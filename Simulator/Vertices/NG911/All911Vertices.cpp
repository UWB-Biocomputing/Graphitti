/**
 * @file All911Vertices.cpp
 * 
 * @ingroup Simulator/Vertices/NG911
 *
 * @brief A container of all 911 vertex data
 */

#include "All911Vertices.h"
#include "All911Edges.h"
#include "GraphManager.h"
#include "Layout911.h"
#include "ParameterManager.h"

All911Vertices::All911Vertices()
{
   callNum_ = nullptr;
   dispNum_ = nullptr;
   respNum_ = nullptr;

   // Get a copy of the file logger to use with log4cplus macros
   fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));
   consoleLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("console"));
}

All911Vertices::~All911Vertices()
{
   if (size_ != 0) {
      delete[] callNum_;
      delete[] dispNum_;
      delete[] respNum_;
   }

   callNum_ = nullptr;
   dispNum_ = nullptr;
   respNum_ = nullptr;
}

// Allocate memory for all class properties
void All911Vertices::setupVertices()
{
   AllVertices::setupVertices();

   callNum_ = new int[size_];
   dispNum_ = new int[size_];
   respNum_ = new int[size_];

   // Populate arrays with 0
   fill_n(callNum_, size_, 0);
   fill_n(dispNum_, size_, 0);
   fill_n(respNum_, size_, 0);

   // Resize and fill vectors with 0
   numAgents_.assign(size_, 0);
   numTrunks_.assign(size_, 0);
   vertexQueues_.resize(size_);

   // Register call properties with InputManager
   inputManager_.registerProperty("vertex_id", &Call::vertexId);
   inputManager_.registerProperty("time", &Call::time);
   inputManager_.registerProperty("duration", &Call::duration);
   inputManager_.registerProperty("x", &Call::x);
   inputManager_.registerProperty("y", &Call::y);
   inputManager_.registerProperty("type", &Call::type);
}

// Generate callNum_ and dispNum_ for all caller and psap nodes
void All911Vertices::createAllVertices(Layout *layout)
{
   // Loop over all vertices and set the number of agents and trunks, and
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
         numAgents_[*vi] = gm[*vi].agents;
         numTrunks_[*vi] = gm[*vi].trunks;

         // The waiting queue is of size # Trunks - # Agents
         int queueSize = numTrunks_[*vi] - numAgents_[*vi];
         vertexQueues_[*vi].resize(queueSize);
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

   Layout911 *layout911 = dynamic_cast<Layout911 *>(layout);

   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      // Create all callers
      if (layout->vertexTypeMap_[i] == CALR) {
         callNum_[i] = initRNG.inRange(callNumRange_[0], callNumRange_[1]);
         callersPerZone[layout911->zone(i)] += callNum_[i];
      }

      // Find all PSAPs
      if (layout->vertexTypeMap_[i] == PSAP) {
         psapList.push_back(i);
      }

      // Find all resps
      if (layout->vertexTypeMap_[i] == RESP) {
         respList.push_back(i);
         respPerZone[layout911->zone(i)] += 1;
      }
   }

   // Create all psaps
   // Dispatchers in a psap = [callers in the zone * k] + some randomness
   for (int i = 0; i < psapList.size(); i++) {
      int psapQ = layout911->zone(i);
      int dispCount = (callersPerZone[psapQ] * dispNumScale_) + initRNG.inRange(-5, 5);
      if (dispCount < 1) {
         dispCount = 1;
      }
      dispNum_[psapList[i]] = dispCount;
   }

   // Create all responders
   // Responders in a node = [callers in the zone * k]/[number of responder nodes] + some randomness
   for (int i = 0; i < respList.size(); i++) {
      int respQ = layout911->zone(respList[i]);
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
}


void All911Vertices::printParameters() const
{
}


string All911Vertices::toString(const int index) const
{
   return nullptr;   // Change this
}


void All911Vertices::loadEpochInputs(uint64_t curStep, uint64_t endStep)
{
   Simulator &simulator = Simulator::getInstance();
   Layout &layout = *(simulator.getModel()->getLayout());

   // Load all the calls into the Caller Regions queue by getting the input events
   // from the InputManager.
   for (int idx = 0; idx < simulator.getTotalVertices(); ++idx) {
      if (layout.vertexTypeMap_[idx] == CALR) {
         // If this is a Caller Region get all calls scheduled for the current epoch
         inputManager_.getEvents(idx, curStep, endStep, vertexQueues_[idx]);
      }
   }
}

#if !defined(USE_GPU)

///  Update internal state of the indexed vertex (called by every simulation step).
///  Notify outgoing edges if vertex has fired.
///
///  @param  edges         The edge list to search from.
///  @param  edgeIndexMap  Reference to the EdgeIndexMap.
void All911Vertices::advanceVertices(AllEdges &edges, const EdgeIndexMap *edgeIndexMap)
{
   Simulator &simulator = Simulator::getInstance();
   Layout &layout = *(simulator.getModel()->getLayout());
   uint64_t endEpochStep
      = g_simulationStep
        + static_cast<uint64_t>(simulator.getEpochDuration() / simulator.getDeltaT());

   All911Edges &edges911 = dynamic_cast<All911Edges &>(edges);

   // Advance vertices
   for (int idx = 0; idx < simulator.getTotalVertices(); ++idx) {
      if (layout.vertexTypeMap_[idx] == CALR) {
         // peek at the next call in the queue
         optional<Call> nextCall = vertexQueues_[idx].peek();
         if (nextCall && nextCall->time == g_simulationStep) {
            // The call starts at the current time step so we need to pop it and process it
            vertexQueues_[idx].get();   // pop from the queue
            LOG4CPLUS_TRACE(consoleLogger_,
                            "Calling PSAP at time: " << nextCall->time);

            // There is only one outgoing edge from CALR to a PSAP
            BGSIZE start = edgeIndexMap->outgoingEdgeBegin_[idx];
            BGSIZE edgeIdx = edgeIndexMap->outgoingEdgeIndexMap_[start];

            // Place new call in the edge going to the PSAP
            if (edges911.isAvailable_[edgeIdx]) {
               edges911.call_[edgeIdx] = nextCall.value();
               edges911.isAvailable_[edgeIdx] = false;
            } else {
               // If the call is still there, it means that there was no space in the PSAP's waiting
               // queue. Therefore, this is a dropped call.
            }
         }
         // TODO911: Check for dropped calls in incoming edge
         
      } else if (layout.vertexTypeMap_[idx] == PSAP) {
         // Get the call from the Waiting queue and print it
         optional<Call> nextCall = vertexQueues_[idx].get();
         if (nextCall) {
            LOG4CPLUS_TRACE(consoleLogger_,
                            "Serving Call starting at time : " << nextCall->time);
         }
      }
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