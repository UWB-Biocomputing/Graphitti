/**
 * @file All911Vertices.cpp
 * 
 * @ingroup Simulator/Vertices/NG911
 *
 * @brief Specialization of the AllVertices class for the NG911 network
 */

#include "All911Vertices.h"
#include "All911Edges.h"
#include "Connections911.h"
#include "GraphManager.h"
#include "Layout911.h"
#include "ParameterManager.h"
#include <cmath>

// Allocate memory for all class properties
void All911Vertices::setupVertices()
{
   AllVertices::setupVertices();

   // Resize and fill vectors with 0
   numServers_.assign(size_, 0);
   busyServers_.assign(size_, 0);
   numTrunks_.assign(size_, 0);
   vertexQueues_.resize(size_);
   servingCall_.resize(size_);
   answerTime_.resize(size_);
   serverCountdown_.resize(size_);

   // Resize and fill data structures for recording
   droppedCalls_.assign(size_, 0);
   receivedCalls_.assign(size_, 0);
   beginTimeHistory_.resize(size_);
   answerTimeHistory_.resize(size_);
   endTimeHistory_.resize(size_);
   wasAbandonedHistory_.resize(size_);
   queueLengthHistory_.resize(size_);
   utilizationHistory_.resize(size_);

   // Register call properties with InputManager
   inputManager_.registerProperty("vertex_id", &Call::vertexId);
   inputManager_.registerProperty("time", &Call::time);
   inputManager_.registerProperty("duration", &Call::duration);
   inputManager_.registerProperty("x", &Call::x);
   inputManager_.registerProperty("y", &Call::y);
   inputManager_.registerProperty("patience", &Call::patience);
   inputManager_.registerProperty("on_site_time", &Call::onSiteTime);
   inputManager_.registerProperty("type", &Call::type);
}


// Creates all the Vertices and assigns initial data for them.
void All911Vertices::createAllVertices(Layout &layout)
{
   // Calcualte the total number of time-steps for the data structures that
   // will record per-step histories
   Simulator &simulator = Simulator::getInstance();
   uint64_t stepsPerEpoch = simulator.getEpochDuration() / simulator.getDeltaT();
   uint64_t totalTimeSteps = stepsPerEpoch * simulator.getNumEpochs();
   BGFLOAT epochDuration = simulator.getEpochDuration();
   BGFLOAT deltaT = simulator.getDeltaT();

   // Loop over all vertices and set the number of servers and trunks, and
   // determine the size of the waiting queue.
   // We get the information needed from the GraphManager.
   GraphManager::VertexIterator vi, vi_end;
   GraphManager &gm = GraphManager::getInstance();
   for (boost::tie(vi, vi_end) = gm.vertices(); vi != vi_end; ++vi) {
      assert(*vi < size_);

      if (gm[*vi].type == "CALR") {
         vertexQueues_[*vi].resize(stepsPerEpoch);
      } else {
         numServers_[*vi] = gm[*vi].servers;
         numTrunks_[*vi] = gm[*vi].trunks;
         // We should not have more servers than trunks
         assert(numServers_[*vi] <= numTrunks_[*vi]);

         // The waiting queue is of size # trunks. We keep track of the # of busy servers
         // to know when there are no more trunks available.
         vertexQueues_[*vi].resize(numTrunks_[*vi]);

         // Initialize the data structures for agent availability
         servingCall_[*vi].resize(gm[*vi].servers);
         answerTime_[*vi].resize(gm[*vi].servers);
         serverCountdown_[*vi].assign(gm[*vi].servers, 0);

         // Initialize the data structures for system metrics
         queueLengthHistory_[*vi].assign(totalTimeSteps, 0);
         utilizationHistory_[*vi].assign(totalTimeSteps, 0);
      }
   }

   // Read Input Events using the InputManager
   inputManager_.readInputs();
}


// Load member variables from configuration file.
void All911Vertices::loadParameters()
{
   ParameterManager::getInstance().getBGFloatByXpath("//RedialP/text()", redialP_);
   ParameterManager::getInstance().getBGFloatByXpath("//AvgDrivingSpeed/text()", avgDrivingSpeed_);
}


// Prints out all parameters of the vertices to logging file.
void All911Vertices::printParameters() const
{
}


string All911Vertices::toString(int index) const
{
   return nullptr;   // Change this
}


// Loads all inputs scheduled to occur in the upcoming epoch.
void All911Vertices::loadEpochInputs(uint64_t currentStep, uint64_t endStep)
{
   Simulator &simulator = Simulator::getInstance();
   Layout &layout = simulator.getModel().getLayout();

   // Load all the calls into the Caller Regions queue by getting the input events
   // from the InputManager.
   for (int idx = 0; idx < simulator.getTotalVertices(); ++idx) {
      if (layout.vertexTypeMap_[idx] == vertexType::CALR) {
         // If this is a Caller Region get all calls scheduled for the current epoch,
         // loading them into the aproppriate index of the vertexQueues_ vector
         inputManager_.getEvents(idx, currentStep, endStep, vertexQueues_[idx]);
      }
   }
}

// Accessor for the waiting queue of a vertex
CircularBuffer<Call> &All911Vertices::getQueue(int vIdx)
{
   return vertexQueues_[vIdx];
}

// Accessor for the droppedCalls counter of a vertex
int &All911Vertices::droppedCalls(int vIdx)
{
   return droppedCalls_[vIdx];
}

// Accessor for the receivedCalls counter of a vertex
int &All911Vertices::receivedCalls(int vIdx)
{
   return receivedCalls_[vIdx];
}

// Accessor for the number of busy servers in a given vertex
int All911Vertices::busyServers(int vIdx) const
{
   return busyServers_[vIdx];
}

#if !defined(USE_GPU)


// Update internal state of the indexed vertex (called by every simulation step).
void All911Vertices::advanceVertices(AllEdges &edges, const EdgeIndexMap &edgeIndexMap)
{
   Simulator &simulator = Simulator::getInstance();
   Layout &layout = simulator.getModel().getLayout();
   uint64_t endEpochStep
      = g_simulationStep
        + static_cast<uint64_t>(simulator.getEpochDuration() / simulator.getDeltaT());

   All911Edges &edges911 = dynamic_cast<All911Edges &>(edges);

   // Advance vertices
   for (int vertex = 0; vertex < simulator.getTotalVertices(); ++vertex) {
      if (layout.vertexTypeMap_[vertex] == vertexType::CALR) {
         advanceCALR(vertex, edges911, edgeIndexMap);
      } else if (layout.vertexTypeMap_[vertex] == vertexType::PSAP) {
         advancePSAP(vertex, edges911, edgeIndexMap);
      } else if (layout.vertexTypeMap_[vertex] == vertexType::EMS || layout.vertexTypeMap_[vertex] == vertexType::FIRE
                 || layout.vertexTypeMap_[vertex] == vertexType::LAW) {
         advanceRESP(vertex, edges911, edgeIndexMap);
      }
   }
}


// Advance a CALR vertex. Send calls to the appropriate PSAP
void All911Vertices::advanceCALR(BGSIZE vertexIdx, All911Edges &edges911,
                                 const EdgeIndexMap &edgeIndexMap)
{
   // There is only one outgoing edge from CALR to a PSAP
   BGSIZE start = edgeIndexMap.outgoingEdgeBegin_[vertexIdx];
   BGSIZE edgeIdx = edgeIndexMap.outgoingEdgeIndexMap_[start];

   // Check for dropped calls, indicated by the edge not being available
   if (!edges911.isAvailable_[edgeIdx]) {
      // If the call is still there, it means that there was no space in the PSAP's waiting
      // queue. Therefore, this is a dropped call.
      // If readialing, we assume that it happens immediately and the caller tries until
      // getting through.
      if (!edges911.isRedial_[edgeIdx] && initRNG.randDblExc() >= redialP_) {
         // We only make the edge available if no readialing occurs.
         edges911.isAvailable_[edgeIdx] = true;
         LOG4CPLUS_DEBUG(vertexLogger_, "Did not redial at time: " << edges911.call_[edgeIdx].time);
      } else {
         // Keep the edge unavailable but mark it as a redial
         edges911.isRedial_[edgeIdx] = true;
      }
   }

   // peek at the next call in the queue
   optional<Call> nextCall = vertexQueues_[vertexIdx].peek();
   if (edges911.isAvailable_[edgeIdx] && nextCall && nextCall->time <= g_simulationStep) {
      // Calls that start at the same time are process in the order they appear.
      // The call starts at the current time step so we need to pop it and process it
      vertexQueues_[vertexIdx].get();   // pop from the queue

      // Place new call in the edge going to the PSAP
      assert(edges911.isAvailable_[edgeIdx]);
      edges911.call_[edgeIdx] = nextCall.value();
      edges911.isAvailable_[edgeIdx] = false;
      LOG4CPLUS_DEBUG(vertexLogger_, "Calling PSAP at time: " << nextCall->time);
   }
}


// Advance a PSAP vertex. Controls the redirection and handling of calls.
void All911Vertices::advancePSAP(BGSIZE vertexIdx, All911Edges &edges911,
                                 const EdgeIndexMap &edgeIndexMap)
{
   // Loop over all servers and free the ones finishing serving calls
   vector<int> availableServers;
   for (size_t server = 0; server < serverCountdown_[vertexIdx].size(); ++server) {
      if (serverCountdown_[vertexIdx][server] == 0) {
         // Server is available to take calls. This check is needed because calls
         // could have duration of zero or server has not been assigned a call yet
         availableServers.push_back(server);
      } else if (--serverCountdown_[vertexIdx][server] == 0) {
         // Server becomes free to take calls
         // TODO: What about wrap-up time?
         Call &endingCall = servingCall_[vertexIdx][server];

         //Store call metrics
         wasAbandonedHistory_[vertexIdx].push_back(false);
         beginTimeHistory_[vertexIdx].push_back(endingCall.time);
         answerTimeHistory_[vertexIdx].push_back(answerTime_[vertexIdx][server]);
         endTimeHistory_[vertexIdx].push_back(g_simulationStep);
         LOG4CPLUS_DEBUG(vertexLogger_,
                         "Finishing call, begin time: "
                            << endingCall.time << ", end time: " << g_simulationStep
                            << ", waited: " << answerTime_[vertexIdx][server] - endingCall.time);

         // Dispatch the Responder closest to the emergency location.
         Connections911 &conn911
            = dynamic_cast<Connections911 &>(Simulator::getInstance().getModel().getConnections());
         BGSIZE respEdge = conn911.getEdgeToClosestResponder(endingCall, vertexIdx);
         BGSIZE responder = edges911.destVertexIndex_[respEdge];
         LOG4CPLUS_DEBUG(vertexLogger_, "Dispatching Responder: " << responder);

         // Place the call in the edge going to the responder
         // Call becomes a dispatch order at this time
         endingCall.time = g_simulationStep;
         edges911.call_[respEdge] = endingCall;
         edges911.isAvailable_[respEdge] = false;

         // This assumes that the caller doesn't stay in the line until the responder
         // arrives on scene. This not true in all instances.
         availableServers.push_back(server);
      }
   }

   // Assign calls to servers until either no servers are available or
   // there are no more calls in the waiting queue
   size_t serverId = 0;
   while (serverId < availableServers.size() && !vertexQueues_[vertexIdx].isEmpty()) {
      // TODO: calls with duration of zero are being added but because countdown will be zero
      //       they don't show up in the logs
      optional<Call> call = vertexQueues_[vertexIdx].get();
      assert(call);

      if (call->patience < (g_simulationStep - call->time)) {
         // If the patience time is less than the waiting time, the call is abandoned
         wasAbandonedHistory_[vertexIdx].push_back(true);
         beginTimeHistory_[vertexIdx].push_back(call->time);
         // Answer time and end time get zero as sentinel for non-valid values
         answerTimeHistory_[vertexIdx].push_back(0);
         endTimeHistory_[vertexIdx].push_back(0);
         LOG4CPLUS_DEBUG(vertexLogger_, "Call was abandoned, Patience: "
                                           << call->patience
                                           << " Ring Time: " << g_simulationStep - call->time);
      } else {
         // The available server starts serving the call
         int availServer = availableServers[serverId];
         servingCall_[vertexIdx][availServer] = call.value();
         answerTime_[vertexIdx][availServer] = g_simulationStep;
         serverCountdown_[vertexIdx][availServer] = call.value().duration;
         LOG4CPLUS_DEBUG(vertexLogger_, "Serving Call starting at time: "
                                           << call->time << ", sim-step: " << g_simulationStep);
         // Next server
         ++serverId;
      }
   }

   // Update number of busy servers. This is used to check if there is space in the queue
   busyServers_[vertexIdx] = numServers_[vertexIdx] - availableServers.size();

   // Update queueLength and utilization histories
   queueLengthHistory_[vertexIdx][g_simulationStep] = vertexQueues_[vertexIdx].size();
   utilizationHistory_[vertexIdx][g_simulationStep]
      = static_cast<double>(busyServers_[vertexIdx]) / numServers_[vertexIdx];
}


// Advance a RESP vertex. Receives call from PSAP and responds to the emergency events
void All911Vertices::advanceRESP(BGSIZE vertexIdx, All911Edges &edges911,
                                 const EdgeIndexMap &edgeIndexMap)
{
   Layout &layout = Simulator::getInstance().getModel().getLayout();

   // Free the units finishing up with emergency responses
   vector<int> availableUnits;
   for (size_t unit = 0; unit < serverCountdown_[vertexIdx].size(); ++unit) {
      if (serverCountdown_[vertexIdx][unit] == 0) {
         // Unit is available
         availableUnits.push_back(unit);
      } else if (--serverCountdown_[vertexIdx][unit] == 0) {
         // Unit becomes available to responde to new incidents
         Call &endingIncident = servingCall_[vertexIdx][unit];

         //Store incident response metrics
         wasAbandonedHistory_[vertexIdx].push_back(false);
         beginTimeHistory_[vertexIdx].push_back(endingIncident.time);
         answerTimeHistory_[vertexIdx].push_back(answerTime_[vertexIdx][unit]);
         endTimeHistory_[vertexIdx].push_back(g_simulationStep);
         LOG4CPLUS_DEBUG(vertexLogger_,
                         "Finishing response, begin time: "
                            << endingIncident.time << ", end time: " << g_simulationStep
                            << ", waited: " << answerTime_[vertexIdx][unit] - endingIncident.time);

         // Unit is added to available units
         availableUnits.push_back(unit);
      }
   }


   // Assign reponse dispatches until no units are available or there are no more
   // incidents in the waiting queue
   for (size_t unit = 0; unit < availableUnits.size() && !vertexQueues_[vertexIdx].isEmpty();
        ++unit) {
      optional<Call> incident = vertexQueues_[vertexIdx].get();
      assert(incident);   // Safety check for valid incidents

      // The available unit starts serving the call
      int availUnit = availableUnits[unit];
      servingCall_[vertexIdx][availUnit] = incident.value();
      answerTime_[vertexIdx][availUnit] = g_simulationStep;

      // We need to calculate the distance in miles but the x and y coordinates
      // represent, respectively, degrees of longitude and latitude.
      // One degree of latitude is aproximately 69 miles regardles of the location. However,
      // a degree of longitude varies, being 69.172 miles at the equator and gradually shrinking
      // to zero at the poles.
      // One degree of longitude can be converted to miles using the following formula:
      //    1 degree of longitude = cos(latitude) * 69.172
      double lngDegreeLength = cos(layout.yloc_[vertexIdx] * (pi / 180)) * 69.172;
      double latDegreeLength = 69.0;
      double deltaLng = incident->x - layout.xloc_[vertexIdx];
      double deltaLat = incident->y - layout.yloc_[vertexIdx];
      double dist2incident
         = sqrt(pow(deltaLng * lngDegreeLength, 2) + pow(deltaLat * latDegreeLength, 2));

      // Calculate the driving time to the incident in seconds
      double driveTime = (dist2incident / avgDrivingSpeed_) * 3600;
      serverCountdown_[vertexIdx][availUnit] = driveTime + incident->onSiteTime;

      serverCountdown_[vertexIdx][availUnit] = incident.value().duration;
      LOG4CPLUS_DEBUG(vertexLogger_, "Response, driving time: " << driveTime << ", On-site time: "
                                                                << incident->onSiteTime);
   }

   // Update number of busy servers. This is used to check if there is space in the queue
   busyServers_[vertexIdx] = numServers_[vertexIdx] - availableUnits.size();

   // Update queueLength and utilization histories
   queueLengthHistory_[vertexIdx][g_simulationStep] = vertexQueues_[vertexIdx].size();
   utilizationHistory_[vertexIdx][g_simulationStep]
      = static_cast<double>(busyServers_[vertexIdx]) / numServers_[vertexIdx];
}
#endif