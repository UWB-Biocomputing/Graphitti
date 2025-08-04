/**
 * @file All911Vertices_d.cpp
 * 
 * @ingroup Simulator/Vertices/NG911
 *
 * @brief Specialization of the AllVertices class for the NG911 network
 */

#include "All911Vertices.h"
#include "All911Edges.h"
#include "Book.h"
#include "Global.h"
#include "GPUModel.h"
#include "InputManager.h"
#include <float.h>

///  CUDA code for advancing all vertices
///
__global__ void advance911VerticesDevice(int totalVertices, 
                                         All911VerticesDeviceProperties *allVerticesDevice, 
                                         All911EdgesDeviceProperties *allEdgesDevice, 
                                         EdgeIndexMapDevice *edgeIndexMapDevice);

/// CUDA code for taking a call from an edge and adding it to a vertex's queue if there is space.
///
__global__ void maybeTakeCallFromEdge(int totalVertices,
                                      int totalNumberOfEvents,
                                      All911VerticesDeviceProperties *allVerticesDevice,
                                      EdgeIndexMapDevice *edgeIndexMapDevice,
                                      All911EdgesDeviceProperties *allEdgesDevice);

void All911Vertices::allocVerticesDeviceStruct()
{
   All911VerticesDeviceProperties allVertices;
   GPUModel *gpuModel = static_cast<GPUModel *>(&Simulator::getInstance().getModel());
   void **allVerticesDevice = reinterpret_cast<void **>(&(gpuModel->getAllVerticesDevice()));
   allocDeviceStruct(allVertices);
   HANDLE_ERROR(cudaMalloc(allVerticesDevice, sizeof(All911VerticesDeviceProperties)));
   HANDLE_ERROR(cudaMemcpy(*allVerticesDevice, &allVertices, sizeof(All911VerticesDeviceProperties),
                           cudaMemcpyHostToDevice));
}

///  Allocate GPU memories to store all vertices' states.
///  (Helper function of allocVerticesDeviceStruct)
///  @param  allVerticesDevice         Reference to the All911VerticesDeviceProperties struct.
void All911Vertices::allocDeviceStruct(All911VerticesDeviceProperties &allVerticesDevice)
{
   Simulator &simulator = Simulator::getInstance();
   uint64_t stepsPerEpoch = simulator.getEpochDuration() / simulator.getDeltaT();
   uint64_t totalTimeSteps = stepsPerEpoch * simulator.getNumEpochs();
   int numberOfVertices = simulator.getTotalVertices();
   int totalNumberOfEvents = inputManager_.getTotalNumberOfEvents();

   //int *vertexType_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.vertexType_, numberOfVertices * sizeof(int)));
   //uint64_t **beginTimeHistory_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.beginTimeHistory_, numberOfVertices * sizeof(uint64_t *)));
   {
      uint64_t *cpuBeginTimeHistory[totalNumberOfEvents];
      for (int i = 0; i < totalNumberOfEvents; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuBeginTimeHistory[i], totalNumberOfEvents * sizeof(uint64_t)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.beginTimeHistory_, cpuBeginTimeHistory,
                              totalNumberOfEvents * sizeof(uint64_t *), cudaMemcpyHostToDevice));
   }
   //uint64_t **answerTimeHistory_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.answerTimeHistory_, numberOfVertices * sizeof(uint64_t *)));
   {
      uint64_t *cpuAnswerTimeHistory[totalNumberOfEvents];
      for (int i = 0; i < totalNumberOfEvents; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuAnswerTimeHistory[i], totalNumberOfEvents * sizeof(uint64_t)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.answerTimeHistory_, cpuAnswerTimeHistory,
                              totalNumberOfEvents * sizeof(uint64_t *), cudaMemcpyHostToDevice));
   }
   //uint64_t **endTimeHistory_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.endTimeHistory_, numberOfVertices * sizeof(uint64_t *)));
   {
      uint64_t *cpuEndTimeHistory[totalNumberOfEvents];
      for (int i = 0; i < totalNumberOfEvents; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuEndTimeHistory[i], totalNumberOfEvents * sizeof(uint64_t)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.endTimeHistory_, cpuEndTimeHistory,
                              totalNumberOfEvents * sizeof(uint64_t *), cudaMemcpyHostToDevice));
   }
   //unsigned char **wasAbandonedHistory_;
   //int **queueLengthHistory_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.queueLengthHistory_, numberOfVertices * sizeof(int *)));
   {
      int *cpuQueueLengthHistory[totalTimeSteps];
      for (int i = 0; i < totalTimeSteps; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuQueueLengthHistory[i], totalTimeSteps * sizeof(int)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.queueLengthHistory_, cpuQueueLengthHistory,
                              totalTimeSteps * sizeof(int *), cudaMemcpyHostToDevice));
   }
   //BGFLOAT **utilizationHistory_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.utilizationHistory_, numberOfVertices * sizeof(int *)));
   {
      int *cpuUtilizationHistory[totalTimeSteps];
      for (int i = 0; i < totalTimeSteps; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuUtilizationHistory[i], totalTimeSteps * sizeof(int)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.queueLengthHistory_, cpuUtilizationHistory,
                              totalTimeSteps * sizeof(int *), cudaMemcpyHostToDevice));
   }
   //int **vertexQueuesBufferVertexId_;
   //uint64_t **vertexQueuesBufferTime_;
   //int **vertexQueuesBufferDuration_;
   //BGFLOAT **vertexQueuesBufferX_;
   //BGFLOAT **vertexQueuesBufferY_;
   //int **vertexQueuesBufferPatience_;
   //int **vertexQueuesBufferOnSiteTime_;
   //int **vertexQueuesBufferResponderType_;
   //uint64_t *vertexQueuesFront_;
   //uint64_t *vertexQueuesEnd_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.droppedCalls_, numberOfVertices * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.receivedCalls_, numberOfVertices * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.busyServers_, numberOfVertices * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.numServers_, numberOfVertices * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.numTrunks_, numberOfVertices * sizeof(int)));
   //int **servingCallBufferVertexId_;
   //uint64_t **servingCallBufferTime_;
   //int **servingCallBufferDuration_;
   //BGFLOAT **servingCallBufferX_;
   //BGFLOAT **servingCallBufferY_;
   //int **servingCallBufferPatience_;
   //int **servingCallBufferOnSiteTime_;
   //int **servingCallBufferResponderType_;
   //uint64_t **answerTime_;
   //int **serverCountdown_;
}

/// @brief Update internal state of the indexed vertex (called by every simulation step).
/// @param edges Reference to the allEdges struct on host memory.
/// @param allVerticesDevice GPU address of the allVerticesDeviceProperties struct on device memory.
/// @param allEdgesDevice GPU address of the allEdgesDeviceProperties struct on device memory.
/// @param randNoise 
/// @param edgeIndexMapDevice GPU address of the EdgeIndexMap on device memory.
void All911Vertices::advanceVertices(AllEdges &edges, void *allVerticesDevice,
                                    void *allEdgesDevice, float randNoise[],
                                    EdgeIndexMapDevice *edgeIndexMapDevice)
{
   // Return if no vertices are present
   if (size_ == 0)
      return;
   // CUDA parameters
   const int threadsPerBlock = 256;
   int blocksPerGrid
      = (Simulator::getInstance().getTotalVertices() + threadsPerBlock - 1) / threadsPerBlock;
   int totalNumberOfEvents = inputManager_.getTotalNumberOfEvents();
   // Advance vertices ------------->
   advance911VerticesDevice<<<blocksPerGrid, threadsPerBlock>>>(size_,
                                                                totalNumberOfEvents,
                                                                g_simulationStep,
                                                                redialP_,
                                                                (All911VerticesDeviceProperties *)allVerticesDevice, 
                                                                (All911EdgesDeviceProperties *)allEdgesDevice, 
                                                                edgeIndexMapDevice);
}

__global__ void advance911VerticesDevice(int totalVertices,
                                         int totalNumberOfEvents,
                                         uint64_t simulationStep,
                                         BGFLOAT redialProbability,
                                         All911VerticesDeviceProperties *allVerticesDevice,
                                         All911EdgesDeviceProperties *allEdgesDevice,
                                         EdgeIndexMapDevice *edgeIndexMapDevice)
{
   // The usual thread ID calculation and guard against excess threads
   // (beyond the number of vertices, in this case).
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx >= totalVertices)
      return;
   
   switch (allVerticesDevice->vertexType_[idx]) {
      case 1: //CALR
         advanceCALRVerticesDevice(idx, totalNumberOfEvents, simulationStep, redialProbability, allEdgesDevice, edgeIndexMapDevice);
         break;
      case 2: //PSAP
         advancePSAPVerticesDevice(idx, totalNumberOfEvents, simulationStep, allVerticesDevice, allEdgesDevice, edgeIndexMapDevice);
         break;
      case 3: //RESP
         advanceRESPVerticesDevice(idx, allEdgesDevice, edgeIndexMapDevice);
         break;
      default:
         assert(false);
   }
}

///  CUDA code for advancing Caller region vertices
///
CUDA_CALLABLE void advanceCALRVerticesDevice(int vertexId,
                                             int totalNumberOfEvents,
                                             uint64_t simulationStep,
                                             BGFLOAT redialProbability, 
                                             All911EdgesDeviceProperties *allEdgesDevice, 
                                             EdgeIndexMapDevice *edgeIndexMapDevice)
{
   // There is only one outgoing edge from CALR to a PSAP
   BGSIZE start = edgeIndexMapDevice->outgoingEdgeBegin_[vertexId];
   BGSIZE edgeIdx = edgeIndexMapDevice->outgoingEdgeIndexMap_[start];

   // Check for dropped calls, indicated by the edge not being available
   if (!allEdgesDevice->isAvailable_[edgeIdx]) {
      // If the call is still there, it means that there was no space in the PSAP's waiting
      // queue. Therefore, this is a dropped call.
      // If readialing, we assume that it happens immediately and the caller tries until
      // getting through.
      if (!allEdgesDevice->isRedial_[edgeIdx] && initRNG.randDblExc() >= redialProbability) {
         // We only make the edge available if no readialing occurs.
         allEdgesDevice->isAvailable_[edgeIdx] = true;
         //LOG4CPLUS_DEBUG(vertexLogger_, "Did not redial at time: " << edges911.call_[edgeIdx].time);
      } else {
         // Keep the edge unavailable but mark it as a redial
         allEdgesDevice->isRedial_[edgeIdx] = true;
      }
   }

   // peek at the next call in the queue
   uint64_t queueEndIndex = allVerticesDevice->vertexQueuesEnd_[vertexId];
   if (allEdgesDevice->isAvailable_[edgeIdx] && 
      (allVerticesDevice->vertexQueuesFront_[vertexId] != queueEndIndex) && 
      allVerticesDevice->vertexQueuesBufferTime_[vertexId][queueEndIndex] <= simulationStep) {
      // Place new call in the edge going to the PSAP
      assert(allEdgesDevice->isAvailable_[edgeIdx]);
      // Calls that start at the same time are process in the order they appear.
      // The call starts at the current time step so we need to pop it and process it

      // Process the call
      allEdgesDevice->vertexId_[edgeIdx] = allVerticesDevice->vertexQueuesBufferVertexId_[vertexId][queueEndIndex];
      allEdgesDevice->time_[edgeIdx] = allVerticesDevice->vertexQueuesBufferTime_[vertexId][queueEndIndex];
      allEdgesDevice->duration_[edgeIdx] = allVerticesDevice->vertexQueuesBufferDuration_[vertexId][queueEndIndex];
      allEdgesDevice->x_[edgeIdx] = allVerticesDevice->vertexQueuesBufferX_[vertexId][queueEndIndex];
      allEdgesDevice->y_[edgeIdx] = allVerticesDevice->vertexQueuesBufferY_[vertexId][queueEndIndex];
      allEdgesDevice->patience_[edgeIdx] = allVerticesDevice->vertexQueuesBufferPatience_[vertexId][queueEndIndex];
      allEdgesDevice->onSiteTime_[edgeIdx] = allVerticesDevice->vertexQueuesBufferOnSiteTime_[vertexId][queueEndIndex];
      allEdgesDevice->responderType_[edgeIdx] = allVerticesDevice->vertexQueuesBufferResponderType_[vertexId][queueEndIndex];

      // Pop from the queue
      allVerticesDevice->vertexQueuesEnd_[vertexId] = (queueEndIndex + 1) % totalNumberOfEvents;
      allEdgesDevice->isAvailable_[edgeIdx] = false;
      //LOG4CPLUS_DEBUG(vertexLogger_, "Calling PSAP at time: " << nextCall->time);
   }
}

///  CUDA code for advancing PSAP vertices
///
CUDA_CALLABLE void advancePSAPVerticesDevice(int vertexId,
                                             int totalNumberOfEvents,
                                             uint64_t simulationStep, 
                                             All911VerticesDeviceProperties *allVerticesDevice, 
                                             All911EdgesDeviceProperties *allEdgesDevice, 
                                             EdgeIndexMapDevice *edgeIndexMapDevice)
{
   // Loop over all servers and free the ones finishing serving calls
   int numberOfServers = allVerticesDevice->numServers_[vertexId];
   unsigned char* availableServers = (unsigned char*) malloc(numberOfServers * sizeof(unsigned char));
   assert(availableServers != nullptr);
   // Initialize to no servers having been assigned a call yet
   for (BGSIZE serverIndex = 0; serverIndex < numberOfServers; serverIndex++) {
      availableServers[serverIndex] = 0;
   }

   for (size_t server = 0; server < numberOfServers; ++server) {
      if (allVerticesDevice->serverCountdown_[vertexIdx][server] == 0) {
         // Server is available to take calls. This check is needed because calls
         // could have duration of zero or server has not been assigned a call yet
         availableServers[server] = 1;
      } else if (--allVerticesDevice->serverCountdown_[vertexIdx][server] == 0) {
         // Server becomes free to take calls
         // TODO: What about wrap-up time?
         //Call &endingCall = servingCall_[vertexIdx][server];

         //Store call metrics
         allVerticesDevice->wasAbandonedHistory_[vertexIdx].push_back(false);
         allVerticesDevice->beginTimeHistory_[vertexIdx].push_back(allVerticesDevice->servingCallBufferTime_[vertexIdx][server]);
         allVerticesDevice->answerTimeHistory_[vertexIdx].push_back(allVerticesDevice->answerTime_[vertexIdx][server]);
         allVerticesDevice->endTimeHistory_[vertexIdx].push_back(simulationStep);
         // LOG4CPLUS_DEBUG(vertexLogger_,
         //                 "Finishing call, begin time: "
         //                    << endingCall.time << ", end time: " << g_simulationStep
         //                    << ", waited: " << answerTime_[vertexIdx][server] - endingCall.time);

         // Dispatch the Responder closest to the emergency location.
         //BGSIZE respEdge = getEdgeToClosestResponder(endingCall, vertexIdx);

         int requiredType = allVerticesDevice->servingCallBufferResponderType_[vertexIdx][server];

         BGSIZE startOutEdg = edgeIndexMapDevice->outgoingEdgeBegin_[vertexIdx];
         BGSIZE outEdgCount = edgeIndexMapDevice->outgoingEdgeCount_[vertexIdx];

         BGSIZE resp, respEdge;
         BGFLOAT minDistance = FLT_MAX;
         for (BGSIZE eIdxMap = startOutEdg; eIdxMap < startOutEdg + outEdgCount; ++eIdxMap) {
            BGSIZE outEdg = edgeIndexMapDevice->outgoingEdgeIndexMap_[eIdxMap];
            assert(allEdgesDevice->inUse_[outEdg]);   // Edge must be in use

            BGSIZE dstVertex = allEdgesDevice->destVertexIndex_[outEdg];
            if (allVerticesDevice->vertexType_[dstVertex] == requiredType) {
               BGFLOAT x = allVerticesDevice->servingCallBufferX_[vertexIdx][server];
               BGFLOAT y = allVerticesDevice->servingCallBufferY_[vertexIdx][server];
               BGFLOAT dstVertexLocationX = allVerticesDevice->vertexLocationX_[dstVertex];
               BGFLOAT dstVertexLocationY = AllVerticesDevice->vertexLocationY_[dstVertex];
               //double distance = layout911.getDistance(dstVertex, call.x, call.y);
               BGFLOAT distance = sqrtf(powf(x - dstVertexLocationX, 2) + (powf(y - dstVertexLocationY, 2)));

               if (distance < minDistance) {
                  minDistance = distance;
                  resp = dstVertex;
                  respEdge = outEdg;
               }
            }
         }

         // We must have found the closest responder of the right type
         assert(minDistance < FLT_MAX);
         assert(allVerticesDevice->vertexType_[respEdge] == requiredType);

         int responder = allEdgesDevice->destVertexIndex_[respEdge];
         // LOG4CPLUS_DEBUG(vertexLogger_, "Dispatching Responder: " << responder);

         // Place the call in the edge going to the responder
         // Call becomes a dispatch order at this time
         allVerticesDevice->servingCallBufferTime_[vertexIdx][server] = simulationStep;
         
         //edges911.call_[respEdge] = endingCall;
         allEdgesDevice->vertexId_[respEdge] = allVerticesDevice->servingCallBufferVertexId_[vertexIdx][server];
         allEdgesDevice->time_[respEdge] = allVerticesDevice->servingCallBufferTime_[vertexIdx][server];
         allEdgesDevice->duration_[respEdge] = allVerticesDevice->servingCallBufferDuration_[vertexIdx][server];
         allEdgesDevice->x_[respEdge] = allVerticesDevice->servingCallBufferX_[vertexIdx][server];
         allEdgesDevice->y_[respEdge] = allVerticesDevice->servingCallBufferY_[vertexIdx][server];
         allEdgesDevice->patience_[respEdge] = allVerticesDevice->servingCallBufferPatience_[vertexIdx][server];
         allEdgesDevice->onSiteTime_[respEdge] = allVerticesDevice->servingCallBufferOnSiteTime_[vertexIdx][server];
         allEdgesDevice->responderType_[respEdge] = allVerticesDevice->servingCallBufferResponderType_[vertexIdx][server];

         allEdgesDevice->isAvailable_[respEdge] = false;

         // This assumes that the caller doesn't stay in the line until the responder
         // arrives on scene. This not true in all instances.
         availableServers[server] = 1;
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

      if (call->patience < (simulationStep - call->time)) {
         // If the patience time is less than the waiting time, the call is abandoned
         wasAbandonedHistory_[vertexIdx].push_back(true);
         beginTimeHistory_[vertexIdx].push_back(call->time);
         // Answer time and end time get zero as sentinel for non-valid values
         answerTimeHistory_[vertexIdx].push_back(0);
         endTimeHistory_[vertexIdx].push_back(0);
         // LOG4CPLUS_DEBUG(vertexLogger_, "Call was abandoned, Patience: "
         //                                   << call->patience
         //                                   << " Ring Time: " << g_simulationStep - call->time);
      } else {
         // The available server starts serving the call
         int availServer = availableServers[serverId];
         servingCall_[vertexIdx][availServer] = call.value();
         allVerticesDevice->answerTime_[vertexIdx][availServer] = simulationStep;
         allVerticesDevice->serverCountdown_[vertexIdx][availServer] = call.value().duration;
         // LOG4CPLUS_DEBUG(vertexLogger_, "Serving Call starting at time: "
         //                                   << call->time << ", sim-step: " << g_simulationStep);
         // Next server
         ++serverId;
      }
   }

   // Update number of busy servers. This is used to check if there is space in the queue
   allVerticesDevice->busyServers_[vertexIdx] = allVerticesDevice->numServers_[vertexIdx] - availableServers.size();

   // Update queueLength and utilization histories
   // Compute the size of the destination queue
   uint64_t queueSize;
   uint64_t queueFrontIndex = allVerticesDevice->vertexQueuesFront_[vertexIdx];
   uint64_t queueEndIndex = allVerticesDevice->vertexQueuesEnd_[vertexIdx];
   if (queueFrontIndex >= queueEndIndex) {
      queueSize = queueFrontIndex - queueEndIndex;
   } else {
      queueSize = totalNumberOfEvents + queueFrontIndex - queueEndIndex;
   }
   allVerticesDevice->queueLengthHistory_[vertexIdx][simulationStep] = queueSize;
   allVerticesDevice->utilizationHistory_[vertexIdx][simulationStep]
      = static_cast<float>(allVerticesDevice->busyServers_[vertexIdx]) / allVerticesDevice->numServers_[vertexIdx];
}

///  CUDA code for advancing emergency responder vertices
///
CUDA_CALLABLE void advanceRESPVerticesDevice(int vertexId, All911EdgesDeviceProperties *allEdgesDevice, EdgeIndexMapDevice *edgeIndexMapDevice)
{

}

/// Take a call from an edge and add it to the queue if the queue isn't full.
///
/// @param allVerticesDevice       GPU address of the allVertices struct on device memory.
/// @param edgeIndexMapDevice      GPU address of the EdgeIndexMap on device memory.
/// @param allEdgesDevice          GPU address of the allEdges struct on device memory.
void All911Vertices::integrateVertexInputs(void *allVerticesDevice,
                                           EdgeIndexMapDevice *edgeIndexMapDevice,
                                           void *allEdgesDevice)
{
   // CUDA parameters
   const int threadsPerBlock = 256;
   int blocksPerGrid
      = (Simulator::getInstance().getTotalVertices() + threadsPerBlock - 1) / threadsPerBlock;
   int totalVertices = Simulator::getInstance().getTotalVertices();
   int totalNumberOfEvents = inputManager_.getTotalNumberOfEvents();
   maybeTakeCallFromEdge<<<blocksPerGrid, threadsPerBlock>>>(totalVertices,
                                                             totalNumberOfEvents, 
                                                             (All911VerticesDeviceProperties *)allVerticesDevice,
                                                             edgeIndexMapDevice, 
                                                             (All911EdgesDeviceProperties *)allEdgesDevice);
}

__global__ void maybeTakeCallFromEdge(int totalVertices,
                                      int totalNumberOfEvents,
                                      All911VerticesDeviceProperties *allVerticesDevice,
                                      EdgeIndexMapDevice *edgeIndexMapDevice,
                                      All911EdgesDeviceProperties *allEdgesDevice)
{
   // The usual thread ID calculation and guard against excess threads
   // (beyond the number of vertices, in this case).
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx >= totalVertices)
      return;
   
   // TODO911: Caller Regions will have different behaviour
   if (allVerticesDevice->vertexType_[idx] == 1)
      return;

   int incomingEdgeStart = edgeIndexMapDevice->incomingEdgeBegin_[idx];
   int incomingEdgeCount = edgeIndexMapDevice->incomingEdgeCount_[idx];

   // Loop over all the edges and pull the data in
   for (int edge = incomingEdgeStart; edge < incomingEdgeStart + incomingEdgeCount; ++edge) {
      int edgeIdx = edgeIndexMapDevice->incomingEdgeIndexMap_[edge];

      if (!allEdgesDevice->inUse_[edgeIdx]) {
         continue;
      }   // Edge isn't in use
      if (allEdgesDevice->isAvailable_[edgeIdx]) {
         continue;
      }   // Edge doesn't have a call

      int dstIndex = allEdgesDevice->destVertexIndex_[edgeIdx];
      // The destination vertex should be the one pulling the information
      assert(dstIndex == idx);

      // Compute the size of the destination queue
      uint64_t dstQueueSize;
      uint64_t queueFrontIndex = allVerticesDevice->vertexQueuesFront_[dstIndex];
      uint64_t queueEndIndex = allVerticesDevice->vertexQueuesEnd_[dstIndex];
      if (queueFrontIndex >= queueEndIndex) {
         dstQueueSize = queueFrontIndex - queueEndIndex;
      } else {
         dstQueueSize = totalNumberOfEvents + queueFrontIndex - queueEndIndex;
      }

      // Compute the capacity of the destination queue
      int dstQueueCapacity = totalNumberOfEvents - 1;

      // Get the number fo busy servers at the destination vertex
      int dstBusyServers = allVerticesDevice->busyServers_[dstIndex];

      if (dstQueueSize >= (dstQueueCapacity - dstBusyServers)) {
         // Call is dropped because there is no space in the waiting queue
         if (!allEdgesDevice->isRedial_[edgeIdx]) {
            // Only count the dropped call if it's not a redial
            allVerticesDevice->droppedCalls_[dstIndex]++;
            // Record that we received a call
            allVerticesDevice->receivedCalls_[dstIndex]++;
            // LOG4CPLUS_DEBUG(vertexLogger_,
            //                   "Call dropped: " << droppedCalls(idx)
            //                                  << ", time: " << all911Edges.call_[edgeIdx].time
            //                                  << ", vertex: " << idx
            //                                  << ", queue size: " << dstQueue.size());
         }
      } else {
         // Transfer call to destination
         // We throw an error if the buffer is full
         assert(!(((queueFrontIndex + 1) % totalNumberOfEvents) == queueEndIndex));
         // Insert the new element and increment the front index
         allVerticesDevice->vertexQueuesBufferVertexId_[dstIndex][queueFrontIndex] = allEdgesDevice->vertexId_[edgeIdx];
         allVerticesDevice->vertexQueuesBufferTime_[dstIndex][queueFrontIndex] = allEdgesDevice->time_[edgeIdx];
         allVerticesDevice->vertexQueuesBufferDuration_[dstIndex][queueFrontIndex] = allEdgesDevice->duration_[edgeIdx];
         allVerticesDevice->vertexQueuesBufferX_[dstIndex][queueFrontIndex] = allEdgesDevice->x_[edgeIdx];
         allVerticesDevice->vertexQueuesBufferY_[dstIndex][queueFrontIndex] = allEdgesDevice->y_[edgeIdx];
         allVerticesDevice->vertexQueuesBufferPatience_[dstIndex][queueFrontIndex] = allEdgesDevice->patience_[edgeIdx];
         allVerticesDevice->vertexQueuesBufferOnSiteTime_[dstIndex][queueFrontIndex] = allEdgesDevice->onSiteTime_[edgeIdx];
         allVerticesDevice->vertexQueuesBufferResponderType_[dstIndex][queueFrontIndex] = allEdgesDevice->responderType_[edgeIdx];
         allVerticesDevice->vertexQueuesFront_[dstIndex] = (queueFrontIndex + 1) % totalNumberOfEvents;
         // Record that we received a call
         allVerticesDevice->receivedCalls_[dstIndex]++;
         allEdgesDevice->isAvailable_[edgeIdx] = true;
         allEdgesDevice->isRedial_[edgeIdx] = false;
      }
   }
}

/// Copies all inputs scheduled to occur in the upcoming epoch onto device.
void All911Vertices::copyEpochInputsToDevice()
{
   GPUModel *gpuModel = static_cast<GPUModel *>(&Simulator::getInstance().getModel());
   void **allVerticesDevice = reinterpret_cast<void **>(&(gpuModel->getAllVerticesDevice()));
   Layout &layout = simulator.getModel().getLayout();

   for (int idx = 0; idx < simulator.getTotalVertices(); ++idx) {
      if (layout.vertexTypeMap_[idx] == vertexType::CALR) {
         // If this is a Caller Region get all calls scheduled for the current epoch,
         // loading them into the aproppriate index of the vertexQueues_ vector
         inputManager_.getEvents(idx, currentStep, endStep, vertexQueues_[idx]);
      }
   }
}