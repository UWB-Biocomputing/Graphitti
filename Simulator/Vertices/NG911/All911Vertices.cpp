/**
 * @file All911Vertices.cpp
 * 
 * @ingroup Simulator/Vertices/NG911
 *
 * @brief A container of all 911 vertex data
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

   callNum_.resize(size_);
   dispNum_.resize(size_);
   respNum_.resize(size_);

   // Populate arrays with 0
   callNum_.assign(size_, 0);
   dispNum_.assign(size_, 0);
   respNum_.assign(size_, 0);

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

   // Register call properties with InputManager
   inputManager_.registerProperty("vertex_id", &Call::vertexId);
   inputManager_.registerProperty("time", &Call::time);
   inputManager_.registerProperty("duration", &Call::duration);
   inputManager_.registerProperty("x", &Call::x);
   inputManager_.registerProperty("y", &Call::y);
   inputManager_.registerProperty("patience", &Call::patience);
   inputManager_.registerProperty("type", &Call::type);
}

// Generate callNum_ and dispNum_ for all caller and psap nodes
void All911Vertices::createAllVertices(Layout &layout)
{
   // Loop over all vertices and set the number of servers and trunks, and
   // determine the size of the waiting queue.
   // We get the information needed from the GraphManager.
   GraphManager::VertexIterator vi, vi_end;
   GraphManager &gm = GraphManager::getInstance();
   for (boost::tie(vi, vi_end) = gm.vertices(); vi != vi_end; ++vi) {
      assert(*vi < size_);

      if (gm[*vi].type == "CALR") {
         // TODO: Hardcoded queue size for now (10/0.0001)
         vertexQueues_[*vi].resize(100000);
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
      }
   }

   // Read Input Events using the InputManager
   inputManager_.readInputs();


   // TODO: The code below is from previous version. I am keeping because
   // it is usefull for testing the output against the previous version.
   vector<int> psapList;
   vector<int> respList;
   psapList.clear();
   respList.clear();

   int callersPerZone[] = {0, 0, 0, 0};
   int respPerZone[] = {0, 0, 0, 0};

   Layout911 &layout911 = dynamic_cast<Layout911 &>(layout);

   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      // Create all callers
      if (layout.vertexTypeMap_[i] == CALR) {
         callNum_[i] = initRNG.inRange(callNumRange_[0], callNumRange_[1]);
         callersPerZone[layout911.zone(i)] += callNum_[i];
      }

      // Find all PSAPs
      if (layout.vertexTypeMap_[i] == PSAP) {
         psapList.push_back(i);
      }

      // Find all resps
      if (layout.vertexTypeMap_[i] == EMS || layout.vertexTypeMap_[i] == FIRE 
          || layout.vertexTypeMap_[i] == LAW) {
         respList.push_back(i);
         respPerZone[layout911.zone(i)] += 1;
      }
   }

   // Create all psaps
   // Dispatchers in a psap = [callers in the zone * k] + some randomness
   for (int i = 0; i < psapList.size(); i++) {
      int psapQ = layout911.zone(i);
      int dispCount = (callersPerZone[psapQ] * dispNumScale_) + initRNG.inRange(-5, 5);
      if (dispCount < 1) {
         dispCount = 1;
      }
      dispNum_[psapList[i]] = dispCount;
   }

   // Create all responders
   // Responders in a node = [callers in the zone * k]/[number of responder nodes] + some randomness
   for (int i = 0; i < respList.size(); i++) {
      int respQ = layout911.zone(respList[i]);
      int respCount
         = (callersPerZone[respQ] * respNumScale_) / respPerZone[respQ] + initRNG.inRange(-5, 5);
      if (respCount < 1) {
         respCount = 1;
      }
      respNum_[respList[i]] = respCount;
   }
}

void All911Vertices::loadParameters()
{
   ParameterManager::getInstance().getIntByXpath("//CallNum/min/text()", callNumRange_[0]);
   ParameterManager::getInstance().getIntByXpath("//CallNum/max/text()", callNumRange_[1]);
   ParameterManager::getInstance().getBGFloatByXpath("//DispNumScale/text()", dispNumScale_);
   ParameterManager::getInstance().getBGFloatByXpath("//RespNumScale/text()", respNumScale_);
   ParameterManager::getInstance().getBGFloatByXpath("//RedialP/text()", redialP_);
}


void All911Vertices::printParameters() const
{
}


string All911Vertices::toString(const int index) const
{
   return nullptr;   // Change this
}


void All911Vertices::loadEpochInputs(uint64_t currentStep, uint64_t endStep)
{
   Simulator &simulator = Simulator::getInstance();
   Layout &layout = simulator.getModel().getLayout();

   // Load all the calls into the Caller Regions queue by getting the input events
   // from the InputManager.
   for (int idx = 0; idx < simulator.getTotalVertices(); ++idx) {
      if (layout.vertexTypeMap_[idx] == CALR) {
         // If this is a Caller Region get all calls scheduled for the current epoch,
         // loading them into the aproppriate index of the vertexQueues_ vector
         inputManager_.getEvents(idx, currentStep, endStep, vertexQueues_[idx]);
      }
   }
}

#if !defined(USE_GPU)

///  Update internal state of the indexed vertex (called by every simulation step).
///  Notify outgoing edges if vertex has fired.
///
///  @param  edges         The edge list to search from.
///  @param  edgeIndexMap  Reference to the EdgeIndexMap.
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
      if (layout.vertexTypeMap_[vertex] == CALR) {
         advanceCALR(vertex, edges911, edgeIndexMap);
      } else if (layout.vertexTypeMap_[vertex] == PSAP) {
         advancePSAP(vertex, edges911, edgeIndexMap);
      } else if (layout.vertexTypeMap_[vertex] == EMS || layout.vertexTypeMap_[vertex] == FIRE 
                 || layout.vertexTypeMap_[vertex] == LAW) {
         advanceRESP(vertex, edges911, edgeIndexMap);
      }
   }
}


void All911Vertices::advanceCALR(const BGSIZE vertexIdx, All911Edges &edges911,
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


void All911Vertices::advancePSAP(const BGSIZE vertexIdx, All911Edges &edges911,
                                 const EdgeIndexMap &edgeIndexMap)
{
   // Loop over all servers and free the ones finishing serving calls
   vector<int> availableServers;
   for (size_t server = 0; server < serverCountdown_[vertexIdx].size(); ++server) {
      if (serverCountdown_[vertexIdx][server] == 0) {
         // Server is available to take calls. This check is needed because
         // calls could have duration of zero, meaning they hang up as soon as
         // the call is answered
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
}


void All911Vertices::advanceRESP(const BGSIZE vertexIdx, All911Edges &edges911,
                                 const EdgeIndexMap &edgeIndexMap)
{
   // Let's just record that responder received the call and pop it from the queue
   if (!vertexQueues_[vertexIdx].isEmpty()) {
      optional<Call> incident = vertexQueues_[vertexIdx].get();
      assert(incident);
      LOG4CPLUS_DEBUG(vertexLogger_, "Responded to incident, type: " << incident->type);
   }
}


///  Update internal state of the indexed Neuron (called by every simulation step).
///
///  @param  index       Index of the Neuron to update.
void All911Vertices::advanceVertex(const int index)
{
   // BGFLOAT &Vm = this->Vm_[index];
   // BGFLOAT &Vthresh = this->Vthresh_[index];
   // BGFLOAT &summationPoint = this->summationMap_[index];
   // BGFLOAT &I0 = this->I0_[index];
   // BGFLOAT &Inoise = this->Inoise_[index];
   // BGFLOAT &C1 = this->C1_[index];
   // BGFLOAT &C2 = this->C2_[index];
   // int &nStepsInRefr = this->numStepsInRefractoryPeriod_[index];

   // if (nStepsInRefr > 0) {
   //     // is neuron refractory?
   //     --nStepsInRefr;
   // } else if (Vm >= Vthresh) {
   //     // should it fire?
   //     fire(index);
   // } else {
   //     summationPoint += I0; // add IO
   //     // add noise
   //     BGFLOAT noise = (*rgNormrnd)();
   //     //LOG4CPLUS_DEBUG(vertexLogger_, "ADVANCE NEURON[" << index << "] :: Noise = " << noise);
   //     summationPoint += noise * Inoise; // add noise
   //     Vm = C1 * Vm + C2 * summationPoint; // decay Vm and add inputs
   // }
   // // clear synaptic input for next time step
   // summationPoint = 0;
}

#endif