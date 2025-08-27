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
#include "Layout.h"
#include "Layout911.h"
#include "Simulator.h"
#include <float.h>
#include <vector>
#include <cstdio>
#include <inttypes.h> //For portable uint64_t formatting in printf

///  CUDA code for advancing all vertices
///
__global__ void advance911VerticesDevice(int totalVertices,
                                         int totalNumberOfEvents,
                                         uint64_t stepsPerEpoch,
                                         uint64_t totalTimeSteps,
                                         uint64_t simulationStep,
                                         BGFLOAT drivingSpeed,
                                         BGFLOAT pi,
                                         BGFLOAT redialProbability,
                                         BGFLOAT *xLocation,
                                         BGFLOAT *yLocation,
                                         All911VerticesDeviceProperties *allVerticesDevice,
                                         All911EdgesDeviceProperties *allEdgesDevice,
                                         EdgeIndexMapDevice *edgeIndexMapDevice);

/// CUDA code for taking a call from an edge and adding it to a vertex's queue if there is space.
///
__global__ void maybeTakeCallFromEdge(int totalVertices,
                                      uint64_t stepsPerEpoch,
                                      All911VerticesDeviceProperties *allVerticesDevice,
                                      All911EdgesDeviceProperties *allEdgesDevice,
                                      EdgeIndexMapDevice *edgeIndexMapDevice);

__device__ void advanceCALRVerticesDevice(int vertexId,
                                             int totalNumberOfEvents,
                                             uint64_t stepsPerEpoch,
                                             uint64_t simulationStep,
                                             BGFLOAT redialValue,
                                             BGFLOAT redialProbability,
                                             All911VerticesDeviceProperties *allVerticesDevice, 
                                             All911EdgesDeviceProperties *allEdgesDevice, 
                                             EdgeIndexMapDevice *edgeIndexMapDevice);

__device__ void advancePSAPVerticesDevice(int vertexIdx,
                                             int totalNumberOfEvents,
                                             uint64_t stepsPerEpoch,
                                             uint64_t totalTimeSteps,
                                             uint64_t simulationStep,
                                             BGFLOAT *xLocation,
                                             BGFLOAT *yLocation,
                                             All911VerticesDeviceProperties *allVerticesDevice, 
                                             All911EdgesDeviceProperties *allEdgesDevice, 
                                             EdgeIndexMapDevice *edgeIndexMapDevice);

__device__ void advanceRESPVerticesDevice(int vertexIdx,
                                             int totalNumberOfEvents,
                                             uint64_t stepsPerEpoch,
                                             uint64_t totalTimeSteps,
                                             uint64_t simulationStep,
                                             BGFLOAT drivingSpeed,
                                             BGFLOAT pi, 
                                             BGFLOAT *xLocation, 
                                             BGFLOAT *yLocation,
                                             All911VerticesDeviceProperties *allVerticesDevice, 
                                             All911EdgesDeviceProperties *allEdgesDevice, 
                                             EdgeIndexMapDevice *edgeIndexMapDevice);

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

   // Layout locations
   Layout &layout = simulator.getModel().getLayout();
   Layout911 &layout911 = dynamic_cast<Layout911 &>(layout);
   layout911.xloc_.allocateDeviceMemory();
   layout911.yloc_.allocateDeviceMemory();

   //int *vertexType_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.vertexType_, numberOfVertices * sizeof(int)));
   // Follow pattern in ALLIFNeurons_d.cpp allocDeviceStruct for spikeHistory to alloc
   // any 2D arrays
   //
   //uint64_t **beginTimeHistory_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.beginTimeHistory_, numberOfVertices * sizeof(uint64_t *)));
   {
      uint64_t *cpuBeginTimeHistory[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuBeginTimeHistory[i], totalNumberOfEvents * sizeof(uint64_t)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.beginTimeHistory_, cpuBeginTimeHistory,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyHostToDevice));
   }
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.beginTimeHistoryBufferFront_, numberOfVertices * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.beginTimeHistoryBufferEnd_, numberOfVertices * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.beginTimeHistoryEpochStart_, numberOfVertices * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.beginTimeHistoryNumElementsInEpoch_, numberOfVertices * sizeof(int)));
   //uint64_t **answerTimeHistory_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.answerTimeHistory_, numberOfVertices * sizeof(uint64_t *)));
   {
      uint64_t *cpuAnswerTimeHistory[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuAnswerTimeHistory[i], totalNumberOfEvents * sizeof(uint64_t)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.answerTimeHistory_, cpuAnswerTimeHistory,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyHostToDevice));
   }
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.answerTimeHistoryBufferFront_, numberOfVertices * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.answerTimeHistoryBufferEnd_, numberOfVertices * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.answerTimeHistoryEpochStart_, numberOfVertices * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.answerTimeHistoryNumElementsInEpoch_, numberOfVertices * sizeof(int)));
   //uint64_t **endTimeHistory_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.endTimeHistory_, numberOfVertices * sizeof(uint64_t *)));
   {
      uint64_t *cpuEndTimeHistory[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuEndTimeHistory[i], totalNumberOfEvents * sizeof(uint64_t)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.endTimeHistory_, cpuEndTimeHistory,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyHostToDevice));
   }
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.endTimeHistoryBufferFront_, numberOfVertices * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.endTimeHistoryBufferEnd_, numberOfVertices * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.endTimeHistoryEpochStart_, numberOfVertices * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.endTimeHistoryNumElementsInEpoch_, numberOfVertices * sizeof(int)));
   //uint64_t **wasAbandonedHistory_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.wasAbandonedHistory_, numberOfVertices * sizeof(uint64_t *)));
   {
      uint64_t *cpuWasAbandonedHistory[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuWasAbandonedHistory[i], totalNumberOfEvents * sizeof(uint64_t)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.wasAbandonedHistory_, cpuWasAbandonedHistory,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyHostToDevice));
   }
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.wasAbandonedHistoryBufferFront_, numberOfVertices * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.wasAbandonedHistoryBufferEnd_, numberOfVertices * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.wasAbandonedHistoryEpochStart_, numberOfVertices * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.wasAbandonedHistoryNumElementsInEpoch_, numberOfVertices * sizeof(int)));
   //uint64_t **queueLengthHistory_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.queueLengthHistory_, numberOfVertices * sizeof(uint64_t *)));
   {
      uint64_t *cpuQueueLengthHistory[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuQueueLengthHistory[i], totalTimeSteps * sizeof(uint64_t)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.queueLengthHistory_, cpuQueueLengthHistory,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyHostToDevice));
   }
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.queueLengthHistoryBufferFront_, numberOfVertices * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.queueLengthHistoryBufferEnd_, numberOfVertices * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.queueLengthHistoryEpochStart_, numberOfVertices * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.queueLengthHistoryNumElementsInEpoch_, numberOfVertices * sizeof(int)));
   //BGFLOAT **utilizationHistory_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.utilizationHistory_, numberOfVertices * sizeof(BGFLOAT *)));
   {
      BGFLOAT *cpuUtilizationHistory[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuUtilizationHistory[i], totalTimeSteps * sizeof(BGFLOAT)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.utilizationHistory_, cpuUtilizationHistory,
                              numberOfVertices * sizeof(BGFLOAT *), cudaMemcpyHostToDevice));
   }
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.utilizationHistoryBufferFront_, numberOfVertices * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.utilizationHistoryBufferEnd_, numberOfVertices * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.utilizationHistoryEpochStart_, numberOfVertices * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.utilizationHistoryNumElementsInEpoch_, numberOfVertices * sizeof(int)));
   //int **vertexQueuesBufferVertexId_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.vertexQueuesBufferVertexId_, numberOfVertices * sizeof(int *)));
   {
      int *cpuCallId[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuCallId[i], stepsPerEpoch * sizeof(int)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.vertexQueuesBufferVertexId_, cpuCallId,
                              numberOfVertices * sizeof(int *), cudaMemcpyHostToDevice));
   }
   //uint64_t **vertexQueuesBufferTime_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.vertexQueuesBufferTime_, numberOfVertices * sizeof(uint64_t *)));
   {
      uint64_t *cpuCallTime[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuCallTime[i], stepsPerEpoch * sizeof(uint64_t)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.vertexQueuesBufferTime_, cpuCallTime,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyHostToDevice));
   }
   //int **vertexQueuesBufferDuration_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.vertexQueuesBufferDuration_, numberOfVertices * sizeof(int *)));
   {
      int *cpuCallDuration[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuCallDuration[i], stepsPerEpoch * sizeof(int)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.vertexQueuesBufferDuration_, cpuCallDuration,
                              numberOfVertices * sizeof(int *), cudaMemcpyHostToDevice));
   }
   //BGFLOAT **vertexQueuesBufferX_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.vertexQueuesBufferX_, numberOfVertices * sizeof(BGFLOAT *)));
   {
      BGFLOAT *cpuCallLocationX[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuCallLocationX[i], stepsPerEpoch * sizeof(BGFLOAT)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.vertexQueuesBufferX_, cpuCallLocationX,
                              numberOfVertices * sizeof(BGFLOAT *), cudaMemcpyHostToDevice));
   }
   //BGFLOAT **vertexQueuesBufferY_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.vertexQueuesBufferY_, numberOfVertices * sizeof(BGFLOAT *)));
   {
      BGFLOAT *cpuCallLocationY[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuCallLocationY[i], stepsPerEpoch * sizeof(BGFLOAT)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.vertexQueuesBufferY_, cpuCallLocationY,
                              numberOfVertices * sizeof(BGFLOAT *), cudaMemcpyHostToDevice));
   }
   //int **vertexQueuesBufferPatience_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.vertexQueuesBufferPatience_, numberOfVertices * sizeof(int *)));
   {
      int *cpuCallPatience[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuCallPatience[i], stepsPerEpoch * sizeof(int)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.vertexQueuesBufferPatience_, cpuCallPatience,
                              numberOfVertices * sizeof(int *), cudaMemcpyHostToDevice));
   }
   //int **vertexQueuesBufferOnSiteTime_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.vertexQueuesBufferOnSiteTime_, numberOfVertices * sizeof(int *)));
   {
      int *cpuCallOnSiteTime[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuCallOnSiteTime[i], stepsPerEpoch * sizeof(int)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.vertexQueuesBufferOnSiteTime_, cpuCallOnSiteTime,
                              numberOfVertices * sizeof(int *), cudaMemcpyHostToDevice));
   }
   //int **vertexQueuesBufferResponderType_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.vertexQueuesBufferResponderType_, numberOfVertices * sizeof(int *)));
   {
      int *cpuCallResponderType[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuCallResponderType[i], stepsPerEpoch * sizeof(int)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.vertexQueuesBufferResponderType_, cpuCallResponderType,
                              numberOfVertices * sizeof(int *), cudaMemcpyHostToDevice));
   }
   //uint64_t *vertexQueuesFront_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.vertexQueuesFront_, numberOfVertices * sizeof(uint64_t)));
   //uint64_t *vertexQueuesEnd_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.vertexQueuesEnd_, numberOfVertices * sizeof(uint64_t)));
   //uint64_t *vertexQueuesBufferSize_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.vertexQueuesBufferSize_, numberOfVertices * sizeof(uint64_t)));
   //int *droppedCalls_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.droppedCalls_, numberOfVertices * sizeof(int)));
   //int *receivedCalls_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.receivedCalls_, numberOfVertices * sizeof(int)));
   //int *busyServers_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.busyServers_, numberOfVertices * sizeof(int)));
   //int *numServers_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.numServers_, numberOfVertices * sizeof(int)));
   //int *numTrunks_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.numTrunks_, numberOfVertices * sizeof(int)));
   //int **servingCallBufferVertexId_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.servingCallBufferVertexId_, numberOfVertices * sizeof(int *)));
   {
      int *cpuCallId[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuCallId[i], maxNumberOfServers_ * sizeof(int)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.servingCallBufferVertexId_, cpuCallId,
                              numberOfVertices * sizeof(int *), cudaMemcpyHostToDevice));
   }
   //uint64_t **servingCallBufferTime_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.servingCallBufferTime_, numberOfVertices * sizeof(uint64_t *)));
   {
      uint64_t *cpuCallTime[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuCallTime[i], maxNumberOfServers_ * sizeof(uint64_t)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.servingCallBufferTime_, cpuCallTime,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyHostToDevice));
   }
   //int **servingCallBufferDuration_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.servingCallBufferDuration_, numberOfVertices * sizeof(int *)));
   {
      int *cpuCallDuration[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuCallDuration[i], maxNumberOfServers_ * sizeof(int)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.servingCallBufferDuration_, cpuCallDuration,
                              numberOfVertices * sizeof(int *), cudaMemcpyHostToDevice));
   }
   //BGFLOAT **servingCallBufferX_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.servingCallBufferX_, numberOfVertices * sizeof(BGFLOAT *)));
   {
      BGFLOAT *cpuCallLocationX[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuCallLocationX[i], maxNumberOfServers_ * sizeof(BGFLOAT)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.servingCallBufferX_, cpuCallLocationX,
                              numberOfVertices * sizeof(BGFLOAT *), cudaMemcpyHostToDevice));
   }
   //BGFLOAT **servingCallBufferY_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.servingCallBufferY_, numberOfVertices * sizeof(BGFLOAT *)));
   {
      BGFLOAT *cpuCallLocationY[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuCallLocationY[i], maxNumberOfServers_ * sizeof(BGFLOAT)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.servingCallBufferY_, cpuCallLocationY,
                              numberOfVertices * sizeof(BGFLOAT *), cudaMemcpyHostToDevice));
   }
   //int **servingCallBufferPatience_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.servingCallBufferPatience_, numberOfVertices * sizeof(int *)));
   {
      int *cpuCallPatience[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuCallPatience[i], maxNumberOfServers_ * sizeof(int)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.servingCallBufferPatience_, cpuCallPatience,
                              numberOfVertices * sizeof(int *), cudaMemcpyHostToDevice));
   }
   //int **servingCallBufferOnSiteTime_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.servingCallBufferOnSiteTime_, numberOfVertices * sizeof(int *)));
   {
      int *cpuCallOnSiteTime[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuCallOnSiteTime[i], maxNumberOfServers_ * sizeof(int)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.servingCallBufferOnSiteTime_, cpuCallOnSiteTime,
                              numberOfVertices * sizeof(int *), cudaMemcpyHostToDevice));
   }
   //int **servingCallBufferResponderType_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.servingCallBufferResponderType_, numberOfVertices * sizeof(int *)));
   {
      int *cpuCallResponderType[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuCallResponderType[i], maxNumberOfServers_ * sizeof(int)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.servingCallBufferResponderType_, cpuCallResponderType,
                              numberOfVertices * sizeof(int *), cudaMemcpyHostToDevice));
   }
   //uint64_t **answerTime_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.answerTime_, numberOfVertices * sizeof(uint64_t *)));
   {
      uint64_t *cpuAnswerTime[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuAnswerTime[i], maxNumberOfServers_ * sizeof(uint64_t)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.answerTime_, cpuAnswerTime,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyHostToDevice));
   }
   //int **serverCountdown_;
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.serverCountdown_, numberOfVertices * sizeof(int *)));
   {
      int *cpuServerCountdown[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
          HANDLE_ERROR(cudaMalloc((void **)&cpuServerCountdown[i], maxNumberOfServers_ * sizeof(int)));
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.serverCountdown_, cpuServerCountdown,
                              numberOfVertices * sizeof(int *), cudaMemcpyHostToDevice));
   }
}

///  Delete GPU memories.
///
void All911Vertices::deleteVerticesDeviceStruct()
{
   All911VerticesDeviceProperties allVertices;
   GPUModel *gpuModel = static_cast<GPUModel *>(&Simulator::getInstance().getModel());
   void *allVerticesDevice = static_cast<void *>(gpuModel->getAllVerticesDevice());
   HANDLE_ERROR(cudaMemcpy(&allVertices, allVerticesDevice,
                           sizeof(All911VerticesDeviceProperties), cudaMemcpyDeviceToHost));
   deleteDeviceStruct(allVertices);
   HANDLE_ERROR(cudaFree(allVerticesDevice));
}

///  Delete GPU memories.
///  (Helper function of deleteVerticesDeviceStruct)
///
///  @param  allVerticesDevice         GPU address of the All911VerticesDeviceProperties struct.
void All911Vertices::deleteDeviceStruct(All911VerticesDeviceProperties &allVerticesDevice)
{
   Simulator &simulator = Simulator::getInstance();
   int numberOfVertices = simulator.getTotalVertices();
   // Free layout locations
   Layout &layout = simulator.getModel().getLayout();
   Layout911 &layout911 = dynamic_cast<Layout911 &>(layout);
   layout911.xloc_.freeDeviceMemory();
   layout911.yloc_.freeDeviceMemory();
   // int *vertexType_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.vertexType_));
   // uint64_t **beginTimeHistory_;
   {
      uint64_t *cpuBeginTimeHistory[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuBeginTimeHistory, allVerticesDevice.beginTimeHistory_,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaFree(cpuBeginTimeHistory[i]));
      }
      HANDLE_ERROR(cudaFree(allVerticesDevice.beginTimeHistory_));
   }
   // int *beginTimeHistoryBufferFront_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.beginTimeHistoryBufferFront_));
   // int *beginTimeHistoryBufferEnd_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.beginTimeHistoryBufferEnd_));
   // int *beginTimeHistoryEpochStart_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.beginTimeHistoryEpochStart_));
   // int *beginTimeHistoryNumElementsInEpoch_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.beginTimeHistoryNumElementsInEpoch_));
   // uint64_t **answerTimeHistory_;
   {
      uint64_t *cpuAnswerTimeHistory[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuAnswerTimeHistory, allVerticesDevice.answerTimeHistory_,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaFree(cpuAnswerTimeHistory[i]));
      }
      HANDLE_ERROR(cudaFree(allVerticesDevice.answerTimeHistory_));
   }
   // int *answerTimeHistoryBufferFront_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.answerTimeHistoryBufferFront_));
   // int *answerTimeHistoryBufferEnd_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.answerTimeHistoryBufferEnd_));
   // int *answerTimeHistoryEpochStart_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.answerTimeHistoryEpochStart_));
   // int *answerTimeHistoryNumElementsInEpoch_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.answerTimeHistoryNumElementsInEpoch_));
   // uint64_t **endTimeHistory_;
   {
      uint64_t *cpuEndTimeHistory[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuEndTimeHistory, allVerticesDevice.endTimeHistory_,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaFree(cpuEndTimeHistory[i]));
      }
      HANDLE_ERROR(cudaFree(allVerticesDevice.endTimeHistory_));
   }
   // int *endTimeHistoryBufferFront_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.endTimeHistoryBufferFront_));
   // int *endTimeHistoryBufferEnd_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.endTimeHistoryBufferEnd_));
   // int *endTimeHistoryEpochStart_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.endTimeHistoryEpochStart_));
   // int *endTimeHistoryNumElementsInEpoch_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.endTimeHistoryNumElementsInEpoch_));
   // uint64_t **wasAbandonedHistory_;
   {
      uint64_t *cpuWasAbandonedHistory[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuWasAbandonedHistory, allVerticesDevice.wasAbandonedHistory_,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaFree(cpuWasAbandonedHistory[i]));
      }
      HANDLE_ERROR(cudaFree(allVerticesDevice.wasAbandonedHistory_));
   }
   // int *wasAbandonedHistoryBufferFront_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.wasAbandonedHistoryBufferFront_));
   // int *wasAbandonedHistoryBufferEnd_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.wasAbandonedHistoryBufferEnd_));
   // int *wasAbandonedHistoryEpochStart_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.wasAbandonedHistoryEpochStart_));
   // int *wasAbandonedHistoryNumElementsInEpoch_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.wasAbandonedHistoryNumElementsInEpoch_));
   // uint64_t **queueLengthHistory_;
   {
      uint64_t *cpuQueueLengthHistory[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuQueueLengthHistory, allVerticesDevice.queueLengthHistory_,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaFree(cpuQueueLengthHistory[i]));
      }
      HANDLE_ERROR(cudaFree(allVerticesDevice.queueLengthHistory_));
   }
   // int *queueLengthHistoryBufferFront_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.queueLengthHistoryBufferFront_));
   // int *queueLengthHistoryBufferEnd_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.queueLengthHistoryBufferEnd_));
   // int *queueLengthHistoryEpochStart_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.queueLengthHistoryEpochStart_));
   // int *queueLengthHistoryNumElementsInEpoch_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.queueLengthHistoryNumElementsInEpoch_));
   // BGFLOAT **utilizationHistory_;
   {
      BGFLOAT *cpuUtilizationHistory[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuUtilizationHistory, allVerticesDevice.utilizationHistory_,
                              numberOfVertices * sizeof(BGFLOAT *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaFree(cpuUtilizationHistory[i]));
      }
      HANDLE_ERROR(cudaFree(allVerticesDevice.utilizationHistory_));
   }
   // int *utilizationHistoryBufferFront_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.utilizationHistoryBufferFront_));
   // int *utilizationHistoryBufferEnd_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.utilizationHistoryBufferEnd_));
   // int *utilizationHistoryEpochStart_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.utilizationHistoryEpochStart_));
   // int *utilizationHistoryNumElementsInEpoch_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.utilizationHistoryNumElementsInEpoch_));
   // int **vertexQueuesBufferVertexId_;
   {
      int *cpuCallId[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuCallId, allVerticesDevice.vertexQueuesBufferVertexId_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaFree(cpuCallId[i]));
      }
      HANDLE_ERROR(cudaFree(allVerticesDevice.vertexQueuesBufferVertexId_));
   }
   // uint64_t **vertexQueuesBufferTime_;
   {
      uint64_t *cpuCallTime[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuCallTime, allVerticesDevice.vertexQueuesBufferTime_,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaFree(cpuCallTime[i]));
      }
      HANDLE_ERROR(cudaFree(allVerticesDevice.vertexQueuesBufferTime_));
   }
   // int **vertexQueuesBufferDuration_;
   {
      int *cpuCallDuration[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuCallDuration, allVerticesDevice.vertexQueuesBufferDuration_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaFree(cpuCallDuration[i]));
      }
      HANDLE_ERROR(cudaFree(allVerticesDevice.vertexQueuesBufferDuration_));
   }
   // BGFLOAT **vertexQueuesBufferX_;
   {
      BGFLOAT *cpuCallLocationX[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuCallLocationX, allVerticesDevice.vertexQueuesBufferX_,
                              numberOfVertices * sizeof(BGFLOAT *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaFree(cpuCallLocationX[i]));
      }
      HANDLE_ERROR(cudaFree(allVerticesDevice.vertexQueuesBufferX_));
   }
   // BGFLOAT **vertexQueuesBufferY_;
   {
      BGFLOAT *cpuCallLocationY[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuCallLocationY, allVerticesDevice.vertexQueuesBufferY_,
                              numberOfVertices * sizeof(BGFLOAT *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaFree(cpuCallLocationY[i]));
      }
      HANDLE_ERROR(cudaFree(allVerticesDevice.vertexQueuesBufferY_));
   }
   // int **vertexQueuesBufferPatience_;
   {
      int *cpuCallPatience[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuCallPatience, allVerticesDevice.vertexQueuesBufferPatience_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaFree(cpuCallPatience[i]));
      }
      HANDLE_ERROR(cudaFree(allVerticesDevice.vertexQueuesBufferPatience_));
   }
   // int **vertexQueuesBufferOnSiteTime_;
   {
      int *cpuCallOnSiteTime[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuCallOnSiteTime, allVerticesDevice.vertexQueuesBufferOnSiteTime_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaFree(cpuCallOnSiteTime[i]));
      }
      HANDLE_ERROR(cudaFree(allVerticesDevice.vertexQueuesBufferOnSiteTime_));
   }
   // int **vertexQueuesBufferResponderType_;
   {
      int *cpuCallResponderType[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuCallResponderType, allVerticesDevice.vertexQueuesBufferResponderType_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaFree(cpuCallResponderType[i]));
      }
      HANDLE_ERROR(cudaFree(allVerticesDevice.vertexQueuesBufferResponderType_));
   }
   // uint64_t *vertexQueuesFront_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.vertexQueuesFront_));
   // uint64_t *vertexQueuesEnd_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.vertexQueuesEnd_));
   // uint64_t *vertexQueuesBufferSize_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.vertexQueuesBufferSize_));
   // int *droppedCalls_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.droppedCalls_));
   // int *receivedCalls_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.receivedCalls_));
   // int *busyServers_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.busyServers_));
   // int *numServers_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.numServers_));
   // int *numTrunks_;
   HANDLE_ERROR(cudaFree(allVerticesDevice.numTrunks_));
   // int **servingCallBufferVertexId_;
   {
      int *cpuCallId[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuCallId, allVerticesDevice.servingCallBufferVertexId_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaFree(cpuCallId[i]));
      }
      HANDLE_ERROR(cudaFree(allVerticesDevice.servingCallBufferVertexId_));
   }
   // uint64_t **servingCallBufferTime_;
   {
      uint64_t *cpuCallTime[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuCallTime, allVerticesDevice.servingCallBufferTime_,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaFree(cpuCallTime[i]));
      }
      HANDLE_ERROR(cudaFree(allVerticesDevice.servingCallBufferTime_));
   }
   // int **servingCallBufferDuration_;
   {
      int *cpuCallDuration[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuCallDuration, allVerticesDevice.servingCallBufferDuration_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaFree(cpuCallDuration[i]));
      }
      HANDLE_ERROR(cudaFree(allVerticesDevice.servingCallBufferDuration_));
   }
   // BGFLOAT **servingCallBufferX_;
   {
      BGFLOAT *cpuCallLocationX[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuCallLocationX, allVerticesDevice.servingCallBufferX_,
                              numberOfVertices * sizeof(BGFLOAT *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaFree(cpuCallLocationX[i]));
      }
      HANDLE_ERROR(cudaFree(allVerticesDevice.servingCallBufferX_));
   }
   // BGFLOAT **servingCallBufferY_;
   {
      BGFLOAT *cpuCallLocationY[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuCallLocationY, allVerticesDevice.servingCallBufferY_,
                              numberOfVertices * sizeof(BGFLOAT *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaFree(cpuCallLocationY[i]));
      }
      HANDLE_ERROR(cudaFree(allVerticesDevice.servingCallBufferY_));
   }
   // int **servingCallBufferPatience_;
   {
      int *cpuCallPatience[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuCallPatience, allVerticesDevice.servingCallBufferPatience_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaFree(cpuCallPatience[i]));
      }
      HANDLE_ERROR(cudaFree(allVerticesDevice.servingCallBufferPatience_));
   }
   // int **servingCallBufferOnSiteTime_;
   {
      int *cpuCallOnSiteTime[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuCallOnSiteTime, allVerticesDevice.servingCallBufferOnSiteTime_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaFree(cpuCallOnSiteTime[i]));
      }
      HANDLE_ERROR(cudaFree(allVerticesDevice.servingCallBufferOnSiteTime_));
   }
   // int **servingCallBufferResponderType_;
   {
      int *cpuCallResponderType[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuCallResponderType, allVerticesDevice.servingCallBufferResponderType_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaFree(cpuCallResponderType[i]));
      }
      HANDLE_ERROR(cudaFree(allVerticesDevice.servingCallBufferResponderType_));
   }
   // uint64_t **answerTime_;
   {
      uint64_t *cpuAnswerTime[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuAnswerTime, allVerticesDevice.answerTime_,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaFree(cpuAnswerTime[i]));
      }
      HANDLE_ERROR(cudaFree(allVerticesDevice.answerTime_));
   }
   // int **serverCountdown_;
   {
      int *cpuServerCountdown[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuServerCountdown, allVerticesDevice.serverCountdown_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaFree(cpuServerCountdown[i]));
      }
      HANDLE_ERROR(cudaFree(allVerticesDevice.serverCountdown_));
   }
}

/// @brief Helper function for copying vertex queues to device from CPU.
/// @pre Memory has been allocated for the All911VerticesDeviceProperties struct. Calls
/// are only of type EMS, FIRE, or LAW.
void All911Vertices::copyVertexQueuesToDevice(int numberOfVertices, uint64_t stepsPerEpoch, All911VerticesDeviceProperties &allVerticesDevice)
{
   // int **vertexQueuesBufferVertexId_;
   {
      int *callIdCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callIdCpu, allVerticesDevice.vertexQueuesBufferVertexId_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<int> callIdInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callIdInBuffer.resize(stepsPerEpoch);
         vector<Call> buffer = vertexQueues_[i].getBuffer();
         for (int j = 0; j < buffer.size(); j++) {
            callIdInBuffer[j] = buffer[j].vertexId;
         }
         HANDLE_ERROR(cudaMemcpy(callIdCpu[i], callIdInBuffer.data(),
                                 stepsPerEpoch * sizeof(int), cudaMemcpyHostToDevice));
         // clear vector before filling with next vertex's call ids
         callIdInBuffer.clear();
      }
   }
   // uint64_t **vertexQueuesBufferTime_;
   {
      uint64_t *callTimeCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callTimeCpu, allVerticesDevice.vertexQueuesBufferTime_,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<uint64_t> callTimeInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callTimeInBuffer.resize(stepsPerEpoch);
         vector<Call> buffer = vertexQueues_[i].getBuffer();
         for (int j = 0; j < buffer.size(); j++) {
            callTimeInBuffer[j] = buffer[j].time;
         }
         HANDLE_ERROR(cudaMemcpy(callTimeCpu[i], callTimeInBuffer.data(),
                                 stepsPerEpoch * sizeof(uint64_t), cudaMemcpyHostToDevice));
         // clear vector before filling with next vertex's call ids
         callTimeInBuffer.clear();
      }
   }
   // int **vertexQueuesBufferDuration_;
   {
      int *callDurationCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callDurationCpu, allVerticesDevice.vertexQueuesBufferDuration_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<int> callDurationInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callDurationInBuffer.resize(stepsPerEpoch);
         vector<Call> buffer = vertexQueues_[i].getBuffer();
         for (int j = 0; j < buffer.size(); j++) {
            callDurationInBuffer[j] = buffer[j].duration;
         }
         HANDLE_ERROR(cudaMemcpy(callDurationCpu[i], callDurationInBuffer.data(),
                                 stepsPerEpoch * sizeof(int), cudaMemcpyHostToDevice));
         // clear vector before filling with next vertex's call ids
         callDurationInBuffer.clear();
      }
   }
   // BGFLOAT **vertexQueuesBufferX_;
   {
      BGFLOAT *callLocationXCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callLocationXCpu, allVerticesDevice.vertexQueuesBufferX_,
                              numberOfVertices * sizeof(BGFLOAT *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<BGFLOAT> callLocationXInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callLocationXInBuffer.resize(stepsPerEpoch);
         vector<Call> buffer = vertexQueues_[i].getBuffer();
         for (int j = 0; j < buffer.size(); j++) {
            callLocationXInBuffer[j] = buffer[j].x;
         }
         HANDLE_ERROR(cudaMemcpy(callLocationXCpu[i], callLocationXInBuffer.data(),
                                 stepsPerEpoch * sizeof(BGFLOAT), cudaMemcpyHostToDevice));
         // clear vector before filling with next vertex's call ids
         callLocationXInBuffer.clear();
      }
   }
   // BGFLOAT **vertexQueuesBufferY_;
   {
      BGFLOAT *callLocationYCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callLocationYCpu, allVerticesDevice.vertexQueuesBufferY_,
                              numberOfVertices * sizeof(BGFLOAT *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<BGFLOAT> callLocationYInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callLocationYInBuffer.resize(stepsPerEpoch);
         vector<Call> buffer = vertexQueues_[i].getBuffer();
         for (int j = 0; j < buffer.size(); j++) {
            callLocationYInBuffer[j] = buffer[j].y;
         }
         HANDLE_ERROR(cudaMemcpy(callLocationYCpu[i], callLocationYInBuffer.data(),
                                 stepsPerEpoch * sizeof(BGFLOAT), cudaMemcpyHostToDevice));
         // clear vector before filling with next vertex's call ids
         callLocationYInBuffer.clear();
      }
   }
   // int **vertexQueuesBufferPatience_;
   {
      int *callPatienceCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callPatienceCpu, allVerticesDevice.vertexQueuesBufferPatience_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<int> callPatienceInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callPatienceInBuffer.resize(stepsPerEpoch);
         vector<Call> buffer = vertexQueues_[i].getBuffer();
         for (int j = 0; j < buffer.size(); j++) {
            callPatienceInBuffer[j] = buffer[j].patience;
         }
         HANDLE_ERROR(cudaMemcpy(callPatienceCpu[i], callPatienceInBuffer.data(),
                                 stepsPerEpoch * sizeof(int), cudaMemcpyHostToDevice));
         // clear vector before filling with next vertex's call ids
         callPatienceInBuffer.clear();
      }
   }
   // int **vertexQueuesBufferOnSiteTime_;
   {
      int *callOnSiteTimeCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callOnSiteTimeCpu, allVerticesDevice.vertexQueuesBufferOnSiteTime_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<int> callOnSiteTimeInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callOnSiteTimeInBuffer.resize(stepsPerEpoch);
         vector<Call> buffer = vertexQueues_[i].getBuffer();
         for (int j = 0; j < buffer.size(); j++) {
            callOnSiteTimeInBuffer[j] = buffer[j].onSiteTime;
         }
         HANDLE_ERROR(cudaMemcpy(callOnSiteTimeCpu[i], callOnSiteTimeInBuffer.data(),
                                 stepsPerEpoch * sizeof(int), cudaMemcpyHostToDevice));
         // clear vector before filling with next vertex's call ids
         callOnSiteTimeInBuffer.clear();
      }
   }
   // int **vertexQueuesBufferResponderType_;
   {
      int *callResponderTypeCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callResponderTypeCpu, allVerticesDevice.vertexQueuesBufferResponderType_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<int> callResponderTypeInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callResponderTypeInBuffer.resize(stepsPerEpoch);
         vector<Call> buffer = vertexQueues_[i].getBuffer();
         for (int j = 0; j < buffer.size(); j++) {
            std::string typeInBuffer = buffer[j].type;
            if (typeInBuffer == "EMS") {
               callResponderTypeInBuffer[j] = 5;
            } else if (typeInBuffer == "Fire") {
               callResponderTypeInBuffer[j] = 6;
            } else if (typeInBuffer == "Law") {
               callResponderTypeInBuffer[j] = 7;
            }
         }
         HANDLE_ERROR(cudaMemcpy(callResponderTypeCpu[i], callResponderTypeInBuffer.data(),
                                 stepsPerEpoch * sizeof(int), cudaMemcpyHostToDevice));
         // clear vector before filling with next vertex's call ids
         callResponderTypeInBuffer.clear();
      }
   }
   // uint64_t *vertexQueuesFront_;
   {
      uint64_t queueFrontCpu[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         queueFrontCpu[i] = vertexQueues_[i].getFrontIndex();
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.vertexQueuesFront_, queueFrontCpu,
                              numberOfVertices * sizeof(uint64_t), cudaMemcpyHostToDevice));
   }
   // uint64_t *vertexQueuesEnd_;
   {
      uint64_t queueEndCpu[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         queueEndCpu[i] = vertexQueues_[i].getEndIndex();
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.vertexQueuesEnd_, queueEndCpu,
                              numberOfVertices * sizeof(uint64_t), cudaMemcpyHostToDevice));
   }
   // uint64_t *vertexQueuesBufferSize_;
   {
      uint64_t queueSizeCpu[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         queueSizeCpu[i] = vertexQueues_[i].getBuffer().size();
      }
      HANDLE_ERROR(cudaMemcpy(allVerticesDevice.vertexQueuesBufferSize_, queueSizeCpu,
                              numberOfVertices * sizeof(uint64_t), cudaMemcpyHostToDevice));
   }
}

/// @brief Helper function for copying serving calls from CPU.
/// @pre Memory has been allocated for the All911VerticesDeviceProperties struct. Calls
/// are only of type EMS, FIRE, or LAW.
void All911Vertices::copyServingCallToDevice(int numberOfVertices, All911VerticesDeviceProperties &allVerticesDevice)
{
   // Logic is similar to copyVertexQueuesToDevice but we use max number of servers
   // for the inner vector dimension
   //
   // int **servingCallBufferVertexId_;
   {
      int *callIdCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callIdCpu, allVerticesDevice.servingCallBufferVertexId_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<int> callIdInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callIdInBuffer.resize(maxNumberOfServers_);
         vector<Call> buffer = servingCall_[i];
         for (int j = 0; j < buffer.size(); j++) {
            callIdInBuffer[j] = buffer[j].vertexId;
         }
         HANDLE_ERROR(cudaMemcpy(callIdCpu[i], callIdInBuffer.data(),
                                 maxNumberOfServers_ * sizeof(int), cudaMemcpyHostToDevice));
         // clear vector before filling with next vertex's call ids
         callIdInBuffer.clear();
      }
   }
   // uint64_t **servingCallBufferTime_;
   {
      uint64_t *callTimeCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callTimeCpu, allVerticesDevice.servingCallBufferTime_,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<uint64_t> callTimeInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callTimeInBuffer.resize(maxNumberOfServers_);
         vector<Call> buffer = servingCall_[i];
         for (int j = 0; j < buffer.size(); j++) {
            callTimeInBuffer[j] = buffer[j].time;
         }
         HANDLE_ERROR(cudaMemcpy(callTimeCpu[i], callTimeInBuffer.data(),
                                 maxNumberOfServers_ * sizeof(uint64_t), cudaMemcpyHostToDevice));
         // clear vector before filling with next vertex's call ids
         callTimeInBuffer.clear();
      }
   }
   // int **servingCallBufferDuration_;
   {
      int *callDurationCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callDurationCpu, allVerticesDevice.servingCallBufferDuration_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<int> callDurationInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callDurationInBuffer.resize(maxNumberOfServers_);
         vector<Call> buffer = servingCall_[i];
         for (int j = 0; j < buffer.size(); j++) {
            callDurationInBuffer[j] = buffer[j].duration;
         }
         HANDLE_ERROR(cudaMemcpy(callDurationCpu[i], callDurationInBuffer.data(),
                                 maxNumberOfServers_ * sizeof(int), cudaMemcpyHostToDevice));
         // clear vector before filling with next vertex's call ids
         callDurationInBuffer.clear();
      }
   }
   // BGFLOAT **servingCallBufferX_;
   {
      BGFLOAT *callLocationXCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callLocationXCpu, allVerticesDevice.servingCallBufferX_,
                              numberOfVertices * sizeof(BGFLOAT *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<BGFLOAT> callLocationXInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callLocationXInBuffer.resize(maxNumberOfServers_);
         vector<Call> buffer = servingCall_[i];
         for (int j = 0; j < buffer.size(); j++) {
            callLocationXInBuffer[j] = buffer[j].x;
         }
         HANDLE_ERROR(cudaMemcpy(callLocationXCpu[i], callLocationXInBuffer.data(),
                                 maxNumberOfServers_ * sizeof(BGFLOAT), cudaMemcpyHostToDevice));
         // clear vector before filling with next vertex's call ids
         callLocationXInBuffer.clear();
      }
   }
   // BGFLOAT **servingCallBufferY_;
   {
      BGFLOAT *callLocationYCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callLocationYCpu, allVerticesDevice.servingCallBufferY_,
                              numberOfVertices * sizeof(BGFLOAT *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<BGFLOAT> callLocationYInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callLocationYInBuffer.resize(maxNumberOfServers_);
         vector<Call> buffer = servingCall_[i];
         for (int j = 0; j < buffer.size(); j++) {
            callLocationYInBuffer[j] = buffer[j].y;
         }
         HANDLE_ERROR(cudaMemcpy(callLocationYCpu[i], callLocationYInBuffer.data(),
                                 maxNumberOfServers_ * sizeof(BGFLOAT), cudaMemcpyHostToDevice));
         // clear vector before filling with next vertex's call ids
         callLocationYInBuffer.clear();
      }
   }
   // int **servingCallBufferPatience_;
   {
      int *callPatienceCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callPatienceCpu, allVerticesDevice.servingCallBufferPatience_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<int> callPatienceInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callPatienceInBuffer.resize(maxNumberOfServers_);
         vector<Call> buffer = servingCall_[i];
         for (int j = 0; j < buffer.size(); j++) {
            callPatienceInBuffer[j] = buffer[j].patience;
         }
         HANDLE_ERROR(cudaMemcpy(callPatienceCpu[i], callPatienceInBuffer.data(),
                                 maxNumberOfServers_ * sizeof(int), cudaMemcpyHostToDevice));
         // clear vector before filling with next vertex's call ids
         callPatienceInBuffer.clear();
      }
   }
   // int **servingCallBufferOnSiteTime_;
   {
      int *callOnSiteTimeCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callOnSiteTimeCpu, allVerticesDevice.servingCallBufferOnSiteTime_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<int> callOnSiteTimeInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callOnSiteTimeInBuffer.resize(maxNumberOfServers_);
         vector<Call> buffer = servingCall_[i];
         for (int j = 0; j < buffer.size(); j++) {
            callOnSiteTimeInBuffer[j] = buffer[j].onSiteTime;
         }
         HANDLE_ERROR(cudaMemcpy(callOnSiteTimeCpu[i], callOnSiteTimeInBuffer.data(),
                                 maxNumberOfServers_ * sizeof(int), cudaMemcpyHostToDevice));
         // clear vector before filling with next vertex's call ids
         callOnSiteTimeInBuffer.clear();
      }
   }
   // int **servingCallBufferResponderType_;
   {
      int *callResponderTypeCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callResponderTypeCpu, allVerticesDevice.servingCallBufferResponderType_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<int> callResponderTypeInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callResponderTypeInBuffer.resize(maxNumberOfServers_);
         vector<Call> buffer = servingCall_[i];
         for (int j = 0; j < buffer.size(); j++) {
            std::string typeInBuffer = buffer[j].type;
            if (typeInBuffer == "EMS") {
               callResponderTypeInBuffer[j] = 5;
            } else if (typeInBuffer == "Fire") {
               callResponderTypeInBuffer[j] = 6;
            } else if (typeInBuffer == "Law") {
               callResponderTypeInBuffer[j] = 7;
            }
         }
         HANDLE_ERROR(cudaMemcpy(callResponderTypeCpu[i], callResponderTypeInBuffer.data(),
                                 maxNumberOfServers_ * sizeof(int), cudaMemcpyHostToDevice));
         // clear vector before filling with next vertex's call ids
         callResponderTypeInBuffer.clear();
      }
   }
}

/// Copy all vertex data from host to device.
void All911Vertices::copyToDevice()
{  
   LOG4CPLUS_DEBUG(vertexLogger_, "Copying All911Vertices to device");
   All911VerticesDeviceProperties allVertices;
   Simulator &simulator = Simulator::getInstance();
   GPUModel *gpuModel = static_cast<GPUModel *>(&(simulator.getModel()));
   void *deviceAddress = static_cast<void *>(gpuModel->getAllVerticesDevice());
   HANDLE_ERROR(cudaMemcpy(&allVertices, deviceAddress,
                           sizeof(All911VerticesDeviceProperties), cudaMemcpyDeviceToHost));

   uint64_t stepsPerEpoch = simulator.getEpochDuration() / simulator.getDeltaT();
   uint64_t totalTimeSteps = stepsPerEpoch * simulator.getNumEpochs();
   int numberOfVertices = simulator.getTotalVertices();
   int totalNumberOfEvents = inputManager_.getTotalNumberOfEvents();

   // Copy layout locations
   Layout &layout = simulator.getModel().getLayout();
   Layout911 &layout911 = dynamic_cast<Layout911 &>(layout);
   layout911.xloc_.copyToDevice();
   layout911.yloc_.copyToDevice();
   // int *vertexType_;
   HANDLE_ERROR(cudaMemcpy(allVertices.vertexType_, vertexType_.data(),
                           numberOfVertices * sizeof(int), cudaMemcpyHostToDevice));
   // uint64_t **beginTimeHistory_;
   {
      uint64_t *cpuBeginTimeHistory[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuBeginTimeHistory, allVertices.beginTimeHistory_,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaMemcpy(cpuBeginTimeHistory[i], beginTimeHistory_[i].data(),
                                 totalNumberOfEvents * sizeof(uint64_t), cudaMemcpyHostToDevice));
      }
   }
   // int *beginTimeHistoryBufferFront_;
   {
      int cpuQueueFront[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         cpuQueueFront[i] = beginTimeHistory_[i].bufferFront_;
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.beginTimeHistoryBufferFront_, cpuQueueFront, numberOfVertices * sizeof(int),
                              cudaMemcpyHostToDevice));
   }
   // int *beginTimeHistoryBufferEnd_;
   {
      int cpuQueueEnd[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         cpuQueueEnd[i] = beginTimeHistory_[i].bufferEnd_;
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.beginTimeHistoryBufferEnd_, cpuQueueEnd, numberOfVertices * sizeof(int),
                              cudaMemcpyHostToDevice));
   }
   // int *beginTimeHistoryEpochStart_;
   {
      int cpuEpochStart[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         cpuEpochStart[i] = beginTimeHistory_[i].epochStart_;
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.beginTimeHistoryEpochStart_, cpuEpochStart, numberOfVertices * sizeof(int),
                              cudaMemcpyHostToDevice));
   }
   // int *beginTimeHistoryNumElementsInEpoch_;
   {
      int cpuElementsInEpoch[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         cpuElementsInEpoch[i] = beginTimeHistory_[i].getNumElementsInEpoch();
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.beginTimeHistoryNumElementsInEpoch_, cpuElementsInEpoch, numberOfVertices * sizeof(int),
                              cudaMemcpyHostToDevice));
   }
   // uint64_t **answerTimeHistory_;
   {
      uint64_t *cpuAnswerTimeHistory[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuAnswerTimeHistory, allVertices.answerTimeHistory_,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaMemcpy(cpuAnswerTimeHistory[i], answerTimeHistory_[i].data(),
                                 totalNumberOfEvents * sizeof(uint64_t), cudaMemcpyHostToDevice));
      }
   }
   // int *answerTimeHistoryBufferFront_;
   {
      int cpuQueueFront[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         cpuQueueFront[i] = answerTimeHistory_[i].bufferFront_;
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.answerTimeHistoryBufferFront_, cpuQueueFront, numberOfVertices * sizeof(int),
                              cudaMemcpyHostToDevice));
   }
   // int *answerTimeHistoryBufferEnd_;
   {
      int cpuQueueEnd[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         cpuQueueEnd[i] = answerTimeHistory_[i].bufferEnd_;
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.answerTimeHistoryBufferEnd_, cpuQueueEnd, numberOfVertices * sizeof(int),
                              cudaMemcpyHostToDevice));
   }
   // int *answerTimeHistoryEpochStart_;
   {
      int cpuEpochStart[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         cpuEpochStart[i] = answerTimeHistory_[i].epochStart_;
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.answerTimeHistoryEpochStart_, cpuEpochStart, numberOfVertices * sizeof(int),
                              cudaMemcpyHostToDevice));
   }
   // int *answerTimeHistoryNumElementsInEpoch_;
   {
      int cpuElementsInEpoch[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         cpuElementsInEpoch[i] = answerTimeHistory_[i].getNumElementsInEpoch();
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.answerTimeHistoryNumElementsInEpoch_, cpuElementsInEpoch, numberOfVertices * sizeof(int),
                              cudaMemcpyHostToDevice));
   }
   // uint64_t **endTimeHistory_;
   {
      uint64_t *cpuEndTimeHistory[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuEndTimeHistory, allVertices.endTimeHistory_,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaMemcpy(cpuEndTimeHistory[i], endTimeHistory_[i].data(),
                                 totalNumberOfEvents * sizeof(uint64_t), cudaMemcpyHostToDevice));
      }
   }
   // int *endTimeHistoryBufferFront_;
   {
      int cpuQueueFront[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         cpuQueueFront[i] = endTimeHistory_[i].bufferFront_;
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.endTimeHistoryBufferFront_, cpuQueueFront, numberOfVertices * sizeof(int),
                              cudaMemcpyHostToDevice));
   }
   // int *endTimeHistoryBufferEnd_;
   {
      int cpuQueueEnd[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         cpuQueueEnd[i] = endTimeHistory_[i].bufferEnd_;
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.endTimeHistoryBufferEnd_, cpuQueueEnd, numberOfVertices * sizeof(int),
                              cudaMemcpyHostToDevice));
   }
   // int *endTimeHistoryEpochStart_;
   {
      int cpuEpochStart[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         cpuEpochStart[i] = endTimeHistory_[i].epochStart_;
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.endTimeHistoryEpochStart_, cpuEpochStart, numberOfVertices * sizeof(int),
                              cudaMemcpyHostToDevice));
   }
   // int *endTimeHistoryNumElementsInEpoch_;
   {
      int cpuElementsInEpoch[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         cpuElementsInEpoch[i] = endTimeHistory_[i].getNumElementsInEpoch();
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.endTimeHistoryNumElementsInEpoch_, cpuElementsInEpoch, numberOfVertices * sizeof(int),
                              cudaMemcpyHostToDevice));
   }
   // uint64_t **wasAbandonedHistory_;
   {
      uint64_t *cpuWasAbandonedHistory[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuWasAbandonedHistory, allVertices.wasAbandonedHistory_,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaMemcpy(cpuWasAbandonedHistory[i], wasAbandonedHistory_[i].data(),
                                 totalNumberOfEvents * sizeof(uint64_t), cudaMemcpyHostToDevice));
      }
   }
   // int *wasAbandonedHistoryBufferFront_;
   {
      int cpuQueueFront[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         cpuQueueFront[i] = wasAbandonedHistory_[i].bufferFront_;
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.wasAbandonedHistoryBufferFront_, cpuQueueFront, numberOfVertices * sizeof(int),
                              cudaMemcpyHostToDevice));
   }
   // int *wasAbandonedHistoryBufferEnd_;
   {
      int cpuQueueEnd[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         cpuQueueEnd[i] = wasAbandonedHistory_[i].bufferEnd_;
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.wasAbandonedHistoryBufferEnd_, cpuQueueEnd, numberOfVertices * sizeof(int),
                              cudaMemcpyHostToDevice));
   }
   // int *wasAbandonedHistoryEpochStart_;
   {
      int cpuEpochStart[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         cpuEpochStart[i] = wasAbandonedHistory_[i].epochStart_;
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.wasAbandonedHistoryEpochStart_, cpuEpochStart, numberOfVertices * sizeof(int),
                              cudaMemcpyHostToDevice));
   }
   // int *wasAbandonedHistoryNumElementsInEpoch_;
   {
      int cpuElementsInEpoch[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         cpuElementsInEpoch[i] = wasAbandonedHistory_[i].getNumElementsInEpoch();
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.wasAbandonedHistoryNumElementsInEpoch_, cpuElementsInEpoch, numberOfVertices * sizeof(int),
                              cudaMemcpyHostToDevice));
   }
   // uint64_t **queueLengthHistory_;
   {
      uint64_t *cpuQueueLengthHistory[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuQueueLengthHistory, allVertices.queueLengthHistory_,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaMemcpy(cpuQueueLengthHistory[i], queueLengthHistory_[i].data(),
                                 totalTimeSteps * sizeof(uint64_t), cudaMemcpyHostToDevice));
      }
   }
   // int *queueLengthHistoryBufferFront_;
   {
      int cpuQueueFront[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         cpuQueueFront[i] = queueLengthHistory_[i].bufferFront_;
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.queueLengthHistoryBufferFront_, cpuQueueFront, numberOfVertices * sizeof(int),
                              cudaMemcpyHostToDevice));
   }
   // int *queueLengthHistoryBufferEnd_;
   {
      int cpuQueueEnd[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         cpuQueueEnd[i] = queueLengthHistory_[i].bufferEnd_;
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.queueLengthHistoryBufferEnd_, cpuQueueEnd, numberOfVertices * sizeof(int),
                              cudaMemcpyHostToDevice));
   }
   // int *queueLengthHistoryEpochStart_;
   {
      int cpuEpochStart[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         cpuEpochStart[i] = queueLengthHistory_[i].epochStart_;
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.queueLengthHistoryEpochStart_, cpuEpochStart, numberOfVertices * sizeof(int),
                              cudaMemcpyHostToDevice));
   }
   // int *queueLengthHistoryNumElementsInEpoch_;
   {
      int cpuElementsInEpoch[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         cpuElementsInEpoch[i] = queueLengthHistory_[i].getNumElementsInEpoch();
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.queueLengthHistoryNumElementsInEpoch_, cpuElementsInEpoch, numberOfVertices * sizeof(int),
                              cudaMemcpyHostToDevice));
   }
   // BGFLOAT **utilizationHistory_;
   {
      BGFLOAT *cpuUtilizationHistory[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuUtilizationHistory, allVertices.utilizationHistory_,
                              numberOfVertices * sizeof(BGFLOAT *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaMemcpy(cpuUtilizationHistory[i], utilizationHistory_[i].data(),
                                 totalTimeSteps * sizeof(BGFLOAT), cudaMemcpyHostToDevice));
      }
   }
   // int *utilizationHistoryBufferFront_;
   {
      int cpuQueueFront[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         cpuQueueFront[i] = utilizationHistory_[i].bufferFront_;
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.utilizationHistoryBufferFront_, cpuQueueFront, numberOfVertices * sizeof(int),
                              cudaMemcpyHostToDevice));
   }
   // int *utilizationHistoryBufferEnd_;
   {
      int cpuQueueEnd[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         cpuQueueEnd[i] = utilizationHistory_[i].bufferEnd_;
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.utilizationHistoryBufferEnd_, cpuQueueEnd, numberOfVertices * sizeof(int),
                              cudaMemcpyHostToDevice));
   }
   // int *utilizationHistoryEpochStart_;
   {
      int cpuEpochStart[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         cpuEpochStart[i] = utilizationHistory_[i].epochStart_;
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.utilizationHistoryEpochStart_, cpuEpochStart, numberOfVertices * sizeof(int),
                              cudaMemcpyHostToDevice));
   }
   // int *utilizationHistoryNumElementsInEpoch_;
   {
      int cpuElementsInEpoch[numberOfVertices];
      for (int i = 0; i < numberOfVertices; i++) {
         cpuElementsInEpoch[i] = utilizationHistory_[i].getNumElementsInEpoch();
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.utilizationHistoryNumElementsInEpoch_, cpuElementsInEpoch, numberOfVertices * sizeof(int),
                              cudaMemcpyHostToDevice));
   }
   // int **vertexQueuesBufferVertexId_;
   // uint64_t **vertexQueuesBufferTime_;
   // int **vertexQueuesBufferDuration_;
   // BGFLOAT **vertexQueuesBufferX_;
   // BGFLOAT **vertexQueuesBufferY_;
   // int **vertexQueuesBufferPatience_;
   // int **vertexQueuesBufferOnSiteTime_;
   // int **vertexQueuesBufferResponderType_;
   // uint64_t *vertexQueuesFront_;
   // uint64_t *vertexQueuesEnd_;
   copyVertexQueuesToDevice(numberOfVertices, stepsPerEpoch, allVertices);
   // int *droppedCalls_;
   HANDLE_ERROR(cudaMemcpy(allVertices.droppedCalls_, droppedCalls_.data(),
                           numberOfVertices * sizeof(int), cudaMemcpyHostToDevice));
   // int *receivedCalls_;
   HANDLE_ERROR(cudaMemcpy(allVertices.receivedCalls_, receivedCalls_.data(),
                           numberOfVertices * sizeof(int), cudaMemcpyHostToDevice));
   // int *busyServers_;
   HANDLE_ERROR(cudaMemcpy(allVertices.busyServers_, busyServers_.data(),
                           numberOfVertices * sizeof(int), cudaMemcpyHostToDevice));
   // int *numServers_;
   HANDLE_ERROR(cudaMemcpy(allVertices.numServers_, numServers_.data(),
                           numberOfVertices * sizeof(int), cudaMemcpyHostToDevice));
   // int *numTrunks_;
   HANDLE_ERROR(cudaMemcpy(allVertices.numTrunks_, numTrunks_.data(),
                           numberOfVertices * sizeof(int), cudaMemcpyHostToDevice));
   // int **servingCallBufferVertexId_;
   // uint64_t **servingCallBufferTime_;
   // int **servingCallBufferDuration_;
   // BGFLOAT **servingCallBufferX_;
   // BGFLOAT **servingCallBufferY_;
   // int **servingCallBufferPatience_;
   // int **servingCallBufferOnSiteTime_;
   // int **servingCallBufferResponderType_;
   copyServingCallToDevice(numberOfVertices, allVertices);
   // uint64_t **answerTime_;
   {
      uint64_t *cpuAnswerTime[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuAnswerTime, allVertices.answerTime_,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaMemcpy(cpuAnswerTime[i], answerTime_[i].data(),
                                 maxNumberOfServers_ * sizeof(uint64_t), cudaMemcpyHostToDevice));
      }
   }
   // int **serverCountdown_;
   {
      int *cpuServerCountdown[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuServerCountdown, allVertices.serverCountdown_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaMemcpy(cpuServerCountdown[i], serverCountdown_[i].data(),
                                 maxNumberOfServers_ * sizeof(int), cudaMemcpyHostToDevice));
      }
   }
}

/// @brief Helper function for copying vertex queues from device to CPU.
/// @pre Memory has been allocated for the All911VerticesDeviceProperties struct. Calls
/// are only of type EMS, FIRE, or LAW.
void All911Vertices::copyVertexQueuesFromDevice(int numberOfVertices, uint64_t stepsPerEpoch, All911VerticesDeviceProperties &allVerticesDevice)
{
   // TODO: Review implementation with Prof Stiber
   // int **vertexQueuesBufferVertexId_;
   {
      int *callIdCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callIdCpu, allVerticesDevice.vertexQueuesBufferVertexId_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<int> callIdInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         // Make sure internal buffer can hold all device values
         callIdInBuffer.resize(stepsPerEpoch);
         HANDLE_ERROR(cudaMemcpy(callIdInBuffer.data(), callIdCpu[i],
                                 stepsPerEpoch * sizeof(int), cudaMemcpyDeviceToHost));
         vector<Call> buffer = vertexQueues_[i].getBuffer();
         // Only copy over the number of IDs that we have on the CPU.
         for (int j = 0; j < buffer.size(); j++) {
            buffer[j].vertexId = callIdInBuffer[j];
         }
         // clear vector before filling with next vertex's call IDs
         callIdInBuffer.clear();
      }
   }
   // uint64_t **vertexQueuesBufferTime_;
   {
      uint64_t *callTimeCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callTimeCpu, allVerticesDevice.vertexQueuesBufferTime_,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<uint64_t> callTimeInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callTimeInBuffer.resize(stepsPerEpoch);
         HANDLE_ERROR(cudaMemcpy(callTimeInBuffer.data(), callTimeCpu[i],
                                 stepsPerEpoch * sizeof(uint64_t), cudaMemcpyDeviceToHost));
         vector<Call> buffer = vertexQueues_[i].getBuffer();
         for (int j = 0; j < buffer.size(); j++) {
            buffer[j].time = callTimeInBuffer[j];
         }
         // clear vector before filling with next vertex's call ids
         callTimeInBuffer.clear();
      }
   }
   // int **vertexQueuesBufferDuration_;
   {
      int *callDurationCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callDurationCpu, allVerticesDevice.vertexQueuesBufferDuration_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<int> callDurationInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callDurationInBuffer.resize(stepsPerEpoch);
         HANDLE_ERROR(cudaMemcpy(callDurationInBuffer.data(), callDurationCpu[i],
                                 stepsPerEpoch * sizeof(int), cudaMemcpyDeviceToHost));
         vector<Call> buffer = vertexQueues_[i].getBuffer();
         for (int j = 0; j < buffer.size(); j++) {
            buffer[j].duration = callDurationInBuffer[j];
         }
         // clear vector before filling with next vertex's call ids
         callDurationInBuffer.clear();
      }
   }
   // BGFLOAT **vertexQueuesBufferX_;
   {
      BGFLOAT *callLocationXCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callLocationXCpu, allVerticesDevice.vertexQueuesBufferX_,
                              numberOfVertices * sizeof(BGFLOAT *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<BGFLOAT> callLocationXInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callLocationXInBuffer.resize(stepsPerEpoch);
         HANDLE_ERROR(cudaMemcpy(callLocationXInBuffer.data(), callLocationXCpu[i],
                                 stepsPerEpoch * sizeof(BGFLOAT), cudaMemcpyDeviceToHost));
         vector<Call> buffer = vertexQueues_[i].getBuffer();
         for (int j = 0; j < buffer.size(); j++) {
            buffer[j].x = callLocationXInBuffer[j];
         }
         // clear vector before filling with next vertex's call ids
         callLocationXInBuffer.clear();
      }
   }
   // BGFLOAT **vertexQueuesBufferY_;
   {
      BGFLOAT *callLocationYCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callLocationYCpu, allVerticesDevice.vertexQueuesBufferY_,
                              numberOfVertices * sizeof(BGFLOAT *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<BGFLOAT> callLocationYInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callLocationYInBuffer.resize(stepsPerEpoch);
         HANDLE_ERROR(cudaMemcpy(callLocationYInBuffer.data(), callLocationYCpu[i],
                                 stepsPerEpoch * sizeof(BGFLOAT), cudaMemcpyDeviceToHost));
         vector<Call> buffer = vertexQueues_[i].getBuffer();
         for (int j = 0; j < buffer.size(); j++) {
            buffer[j].y = callLocationYInBuffer[j];
         }
         // clear vector before filling with next vertex's call ids
         callLocationYInBuffer.clear();
      }
   }
   // int **vertexQueuesBufferPatience_;
   {
      int *callPatienceCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callPatienceCpu, allVerticesDevice.vertexQueuesBufferPatience_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<int> callPatienceInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callPatienceInBuffer.resize(stepsPerEpoch);
         HANDLE_ERROR(cudaMemcpy(callPatienceInBuffer.data(), callPatienceCpu[i],
                                 stepsPerEpoch * sizeof(int), cudaMemcpyDeviceToHost));
         vector<Call> buffer = vertexQueues_[i].getBuffer();
         for (int j = 0; j < buffer.size(); j++) {
            buffer[j].patience = callPatienceInBuffer[j];
         }
         // clear vector before filling with next vertex's call ids
         callPatienceInBuffer.clear();
      }
   }
   // int **vertexQueuesBufferOnSiteTime_;
   {
      int *callOnSiteTimeCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callOnSiteTimeCpu, allVerticesDevice.vertexQueuesBufferOnSiteTime_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<int> callOnSiteTimeInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callOnSiteTimeInBuffer.resize(stepsPerEpoch);
         HANDLE_ERROR(cudaMemcpy(callOnSiteTimeInBuffer.data(), callOnSiteTimeCpu[i],
                                 stepsPerEpoch * sizeof(int), cudaMemcpyDeviceToHost));
         vector<Call> buffer = vertexQueues_[i].getBuffer();
         for (int j = 0; j < buffer.size(); j++) {
            buffer[j].onSiteTime = callOnSiteTimeInBuffer[j];
         }
         // clear vector before filling with next vertex's call ids
         callOnSiteTimeInBuffer.clear();
      }
   }
   // int **vertexQueuesBufferResponderType_;
   {
      int *callResponderTypeCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callResponderTypeCpu, allVerticesDevice.vertexQueuesBufferResponderType_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<int> callResponderTypeInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callResponderTypeInBuffer.resize(stepsPerEpoch);
         HANDLE_ERROR(cudaMemcpy(callResponderTypeInBuffer.data(), callResponderTypeCpu[i],
                                 stepsPerEpoch * sizeof(int), cudaMemcpyDeviceToHost));
         vector<Call> buffer = vertexQueues_[i].getBuffer();
         for (int j = 0; j < buffer.size(); j++) {
            if (callResponderTypeInBuffer[j] == 5) {
               buffer[j].type = "EMS";
            } else if (callResponderTypeInBuffer[j] == 6) {
               buffer[j].type = "Fire";
            } else if (callResponderTypeInBuffer[j] == 7) {
               buffer[j].type = "Law";
            }
         }
         // clear vector before filling with next vertex's call ids
         callResponderTypeInBuffer.clear();
      }
   }
   // uint64_t *vertexQueuesFront_;
   {
      uint64_t queueFrontCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(queueFrontCpu, allVerticesDevice.vertexQueuesFront_,
                              numberOfVertices * sizeof(uint64_t), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         vertexQueues_[i].setFrontIndex(queueFrontCpu[i]);
      }
   }
   // uint64_t *vertexQueuesEnd_;
   {
      uint64_t queueEndCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(queueEndCpu, allVerticesDevice.vertexQueuesEnd_,
                              numberOfVertices * sizeof(uint64_t), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         vertexQueues_[i].setEndIndex(queueEndCpu[i]);
      }
   }
   // uint64_t *vertexQueuesBufferSize_;
   {
      uint64_t queueSizeCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(queueSizeCpu, allVerticesDevice.vertexQueuesBufferSize_,
                              numberOfVertices * sizeof(uint64_t), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         vertexQueues_[i].getBuffer().resize(queueSizeCpu[i]);
      }
   }
}

void All911Vertices::copyServingCallFromDevice(int numberOfVertices, All911VerticesDeviceProperties &allVerticesDevice)
{
   // int **servingCallBufferVertexId_;
   {
      int *callIdCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callIdCpu, allVerticesDevice.servingCallBufferVertexId_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<int> callIdInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callIdInBuffer.resize(maxNumberOfServers_);
         HANDLE_ERROR(cudaMemcpy(callIdInBuffer.data(), callIdCpu[i],
                                 maxNumberOfServers_ * sizeof(int), cudaMemcpyDeviceToHost));
         vector<Call> buffer = servingCall_[i];
         for (int j = 0; j < buffer.size(); j++) {
            buffer[j].vertexId = callIdInBuffer[j];
         }
         // clear vector before filling with next vertex's call ids
         callIdInBuffer.clear();
      }
   }
   // uint64_t **servingCallBufferTime_;
   {
      uint64_t *callTimeCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callTimeCpu, allVerticesDevice.servingCallBufferTime_,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<uint64_t> callTimeInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callTimeInBuffer.resize(maxNumberOfServers_);
         HANDLE_ERROR(cudaMemcpy(callTimeInBuffer.data(), callTimeCpu[i],
                                 maxNumberOfServers_ * sizeof(uint64_t), cudaMemcpyDeviceToHost));
         vector<Call> buffer = servingCall_[i];
         for (int j = 0; j < buffer.size(); j++) {
            buffer[j].time = callTimeInBuffer[j];
         }
         // clear vector before filling with next vertex's call ids
         callTimeInBuffer.clear();
      }
   }
   // int **servingCallBufferDuration_;
   {
      int *callDurationCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callDurationCpu, allVerticesDevice.servingCallBufferDuration_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<int> callDurationInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callDurationInBuffer.resize(maxNumberOfServers_);
         HANDLE_ERROR(cudaMemcpy(callDurationInBuffer.data(), callDurationCpu[i],
                                 maxNumberOfServers_ * sizeof(int), cudaMemcpyDeviceToHost));
         vector<Call> buffer = servingCall_[i];
         for (int j = 0; j < buffer.size(); j++) {
            buffer[j].duration = callDurationInBuffer[j];
         }
         // clear vector before filling with next vertex's call ids
         callDurationInBuffer.clear();
      }
   }
   // BGFLOAT **servingCallBufferX_;
   {
      BGFLOAT *callLocationXCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callLocationXCpu, allVerticesDevice.servingCallBufferX_,
                              numberOfVertices * sizeof(BGFLOAT *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<BGFLOAT> callLocationXInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callLocationXInBuffer.resize(maxNumberOfServers_);
         HANDLE_ERROR(cudaMemcpy(callLocationXInBuffer.data(), callLocationXCpu[i],
                                 maxNumberOfServers_ * sizeof(BGFLOAT), cudaMemcpyDeviceToHost));
         vector<Call> buffer = servingCall_[i];
         for (int j = 0; j < buffer.size(); j++) {
            buffer[j].x = callLocationXInBuffer[j];
         }
         // clear vector before filling with next vertex's call ids
         callLocationXInBuffer.clear();
      }
   }
   // BGFLOAT **servingCallBufferY_;
   {
      BGFLOAT *callLocationYCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callLocationYCpu, allVerticesDevice.servingCallBufferY_,
                              numberOfVertices * sizeof(BGFLOAT *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<BGFLOAT> callLocationYInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callLocationYInBuffer.resize(maxNumberOfServers_);
         HANDLE_ERROR(cudaMemcpy(callLocationYInBuffer.data(), callLocationYCpu[i],
                                 maxNumberOfServers_ * sizeof(BGFLOAT), cudaMemcpyDeviceToHost));
         vector<Call> buffer = servingCall_[i];
         for (int j = 0; j < buffer.size(); j++) {
            buffer[j].y = callLocationYInBuffer[j];
         }
         // clear vector before filling with next vertex's call ids
         callLocationYInBuffer.clear();
      }
   }
   // int **servingCallBufferPatience_;
   {
      int *callPatienceCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callPatienceCpu, allVerticesDevice.servingCallBufferPatience_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<int> callPatienceInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callPatienceInBuffer.resize(maxNumberOfServers_);
         HANDLE_ERROR(cudaMemcpy(callPatienceInBuffer.data(), callPatienceCpu[i],
                                 maxNumberOfServers_ * sizeof(int), cudaMemcpyDeviceToHost));
         vector<Call> buffer = servingCall_[i];
         for (int j = 0; j < buffer.size(); j++) {
            buffer[j].patience = callPatienceInBuffer[j];
         }
         // clear vector before filling with next vertex's call ids
         callPatienceInBuffer.clear();
      }
   }
   // int **servingCallBufferOnSiteTime_;
   {
      int *callOnSiteTimeCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callOnSiteTimeCpu, allVerticesDevice.servingCallBufferOnSiteTime_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<int> callOnSiteTimeInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callOnSiteTimeInBuffer.resize(maxNumberOfServers_);
         HANDLE_ERROR(cudaMemcpy(callOnSiteTimeInBuffer.data(), callOnSiteTimeCpu[i],
                                 maxNumberOfServers_ * sizeof(int), cudaMemcpyDeviceToHost));
         vector<Call> buffer = servingCall_[i];
         for (int j = 0; j < buffer.size(); j++) {
            buffer[j].onSiteTime = callOnSiteTimeInBuffer[j];
         }
         // clear vector before filling with next vertex's call ids
         callOnSiteTimeInBuffer.clear();
      }
   }
   // int **servingCallBufferResponderType_;
   {
      int *callResponderTypeCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callResponderTypeCpu, allVerticesDevice.servingCallBufferResponderType_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<int> callResponderTypeInBuffer;
      for (int i = 0; i < numberOfVertices; i++) {
         callResponderTypeInBuffer.resize(maxNumberOfServers_);
         HANDLE_ERROR(cudaMemcpy(callResponderTypeInBuffer.data(), callResponderTypeCpu[i],
                                 maxNumberOfServers_ * sizeof(int), cudaMemcpyDeviceToHost));
         vector<Call> buffer = servingCall_[i];
         for (int j = 0; j < buffer.size(); j++) {
            if (callResponderTypeInBuffer[j] == 5) {
               buffer[j].type = "EMS";
            } else if (callResponderTypeInBuffer[j] == 6) {
               buffer[j].type = "Fire";
            } else if (callResponderTypeInBuffer[j] == 7) {
               buffer[j].type = "Law";
            }
         }
         // clear vector before filling with next vertex's call ids
         callResponderTypeInBuffer.clear();
      }
   }
}

/// Copy all vertex data to host from device.
void All911Vertices::copyFromDevice()
{
   All911VerticesDeviceProperties allVertices;
   Simulator &simulator = Simulator::getInstance();
   GPUModel *gpuModel = static_cast<GPUModel *>(&(simulator.getModel()));
   void *deviceAddress = static_cast<void *>(gpuModel->getAllVerticesDevice());
   HANDLE_ERROR(cudaMemcpy(&allVertices, deviceAddress,
                           sizeof(All911VerticesDeviceProperties), cudaMemcpyDeviceToHost));                     

   uint64_t stepsPerEpoch = simulator.getEpochDuration() / simulator.getDeltaT();
   uint64_t totalTimeSteps = stepsPerEpoch * simulator.getNumEpochs();
   int numberOfVertices = simulator.getTotalVertices();
   int totalNumberOfEvents = inputManager_.getTotalNumberOfEvents();

   // Copy layout locations
   Layout &layout = simulator.getModel().getLayout();
   Layout911 &layout911 = dynamic_cast<Layout911 &>(layout);
   layout911.xloc_.copyToHost();
   layout911.yloc_.copyToHost();
   // int *vertexType_;
   HANDLE_ERROR(cudaMemcpy(vertexType_.data(), allVertices.vertexType_, numberOfVertices * sizeof(int),
                           cudaMemcpyDeviceToHost));
   // uint64_t **beginTimeHistory_;
   {
      uint64_t *cpuBeginTimeHistory[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuBeginTimeHistory, allVertices.beginTimeHistory_,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaMemcpy(beginTimeHistory_[i].data(), cpuBeginTimeHistory[i],
                                 totalNumberOfEvents * sizeof(uint64_t), cudaMemcpyDeviceToHost));
      }
   }
   // int *beginTimeHistoryBufferFront_;
   {
      int cpuQueueFront[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuQueueFront, allVertices.beginTimeHistoryBufferFront_, numberOfVertices * sizeof(int),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         beginTimeHistory_[i].bufferFront_ = cpuQueueFront[i];
      }
   }
   // int *beginTimeHistoryBufferEnd_;
   {
      int cpuQueueEnd[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuQueueEnd, allVertices.beginTimeHistoryBufferEnd_, numberOfVertices * sizeof(int),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         beginTimeHistory_[i].bufferEnd_ = cpuQueueEnd[i];
      }
   }
   // int *beginTimeHistoryEpochStart_;
   {
      int cpuEpochStart[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuEpochStart, allVertices.beginTimeHistoryEpochStart_, numberOfVertices * sizeof(int),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         beginTimeHistory_[i].epochStart_ = cpuEpochStart[i];
      }
   }
   // int *beginTimeHistoryNumElementsInEpoch_;
   {
      int cpuElementsInEpoch[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuElementsInEpoch, allVertices.beginTimeHistoryNumElementsInEpoch_, numberOfVertices * sizeof(int),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         beginTimeHistory_[i].numElementsInEpoch_ = cpuElementsInEpoch[i];
      }
   }
   // uint64_t **answerTimeHistory_;
   {
      uint64_t *cpuAnswerTimeHistory[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuAnswerTimeHistory, allVertices.answerTimeHistory_,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaMemcpy(answerTimeHistory_[i].data(), cpuAnswerTimeHistory[i],
                                 totalNumberOfEvents * sizeof(uint64_t), cudaMemcpyDeviceToHost));
      }
   }
   // int *answerTimeHistoryBufferFront_;
   {
      int cpuQueueFront[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuQueueFront, allVertices.answerTimeHistoryBufferFront_, numberOfVertices * sizeof(int),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         answerTimeHistory_[i].bufferFront_ = cpuQueueFront[i];
      }
   }
   // int *answerTimeHistoryBufferEnd_;
   {
      int cpuQueueEnd[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuQueueEnd, allVertices.answerTimeHistoryBufferEnd_, numberOfVertices * sizeof(int),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         answerTimeHistory_[i].bufferEnd_ = cpuQueueEnd[i];
      }
   }
   // int *answerTimeHistoryEpochStart_;
   {
      int cpuEpochStart[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuEpochStart, allVertices.answerTimeHistoryEpochStart_, numberOfVertices * sizeof(int),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         answerTimeHistory_[i].epochStart_ = cpuEpochStart[i];
      }
   }
   // int *answerTimeHistoryNumElementsInEpoch_;
   {
      int cpuElementsInEpoch[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuElementsInEpoch, allVertices.answerTimeHistoryNumElementsInEpoch_, numberOfVertices * sizeof(int),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         answerTimeHistory_[i].numElementsInEpoch_ = cpuElementsInEpoch[i];
      }
   }
   // uint64_t **endTimeHistory_;
   {
      uint64_t *cpuEndTimeHistory[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuEndTimeHistory, allVertices.endTimeHistory_,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaMemcpy(endTimeHistory_[i].data(), cpuEndTimeHistory[i],
                                 totalNumberOfEvents * sizeof(uint64_t), cudaMemcpyDeviceToHost));
      }
   }
   // int *endTimeHistoryBufferFront_;
   {
      int cpuQueueFront[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuQueueFront, allVertices.endTimeHistoryBufferFront_, numberOfVertices * sizeof(int),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         endTimeHistory_[i].bufferFront_ = cpuQueueFront[i];
      }
   }
   // int *endTimeHistoryBufferEnd_;
   {
      int cpuQueueEnd[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuQueueEnd, allVertices.endTimeHistoryBufferEnd_, numberOfVertices * sizeof(int),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         endTimeHistory_[i].bufferEnd_ = cpuQueueEnd[i];
      }
   }
   // int *endTimeHistoryEpochStart_;
   {
      int cpuEpochStart[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuEpochStart, allVertices.endTimeHistoryEpochStart_, numberOfVertices * sizeof(int),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         endTimeHistory_[i].epochStart_ = cpuEpochStart[i];
      }
   }
   // int *endTimeHistoryNumElementsInEpoch_;
   {
      int cpuElementsInEpoch[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuElementsInEpoch, allVertices.endTimeHistoryNumElementsInEpoch_, numberOfVertices * sizeof(int),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         endTimeHistory_[i].numElementsInEpoch_ = cpuElementsInEpoch[i];
      }
   }
   // uint64_t **wasAbandonedHistory_;
   {
      uint64_t *cpuWasAbandonedHistory[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuWasAbandonedHistory, allVertices.wasAbandonedHistory_,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaMemcpy(wasAbandonedHistory_[i].data(), cpuWasAbandonedHistory[i],
                                 totalNumberOfEvents * sizeof(uint64_t), cudaMemcpyDeviceToHost));
      }
   }
   // int *wasAbandonedHistoryBufferFront_;
   {
      int cpuQueueFront[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuQueueFront, allVertices.wasAbandonedHistoryBufferFront_, numberOfVertices * sizeof(int),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         wasAbandonedHistory_[i].bufferFront_ = cpuQueueFront[i];
      }
   }
   // int *wasAbandonedHistoryBufferEnd_;
   {
      int cpuQueueEnd[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuQueueEnd, allVertices.wasAbandonedHistoryBufferEnd_, numberOfVertices * sizeof(int),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         wasAbandonedHistory_[i].bufferEnd_ = cpuQueueEnd[i];
      }
   }
   // int *wasAbandonedHistoryEpochStart_;
   {
      int cpuEpochStart[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuEpochStart, allVertices.wasAbandonedHistoryEpochStart_, numberOfVertices * sizeof(int),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         wasAbandonedHistory_[i].epochStart_ = cpuEpochStart[i];
      }
   }
   // int *wasAbandonedHistoryNumElementsInEpoch_;
   {
      int cpuElementsInEpoch[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuElementsInEpoch, allVertices.wasAbandonedHistoryNumElementsInEpoch_, numberOfVertices * sizeof(int),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         wasAbandonedHistory_[i].numElementsInEpoch_ = cpuElementsInEpoch[i];
      }
   }
   // uint64_t **queueLengthHistory_;
   {
      uint64_t *cpuQueueLengthHistory[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuQueueLengthHistory, allVertices.queueLengthHistory_,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaMemcpy(queueLengthHistory_[i].data(), cpuQueueLengthHistory[i],
                                 totalTimeSteps * sizeof(uint64_t), cudaMemcpyDeviceToHost));
      }
   }
   // int *queueLengthHistoryBufferFront_;
   {
      int cpuQueueFront[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuQueueFront, allVertices.queueLengthHistoryBufferFront_, numberOfVertices * sizeof(int),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         queueLengthHistory_[i].bufferFront_ = cpuQueueFront[i];
      }
   }
   // int *queueLengthHistoryBufferEnd_;
   {
      int cpuQueueEnd[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuQueueEnd, allVertices.queueLengthHistoryBufferEnd_, numberOfVertices * sizeof(int),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         queueLengthHistory_[i].bufferEnd_ = cpuQueueEnd[i];
      }
   }
   // int *queueLengthHistoryEpochStart_;
   {
      int cpuEpochStart[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuEpochStart, allVertices.queueLengthHistoryEpochStart_, numberOfVertices * sizeof(int),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         queueLengthHistory_[i].epochStart_ = cpuEpochStart[i];
      }
   }
   // int *queueLengthHistoryNumElementsInEpoch_;
   {
      int cpuElementsInEpoch[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuElementsInEpoch, allVertices.queueLengthHistoryNumElementsInEpoch_, numberOfVertices * sizeof(int),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         queueLengthHistory_[i].numElementsInEpoch_ = cpuElementsInEpoch[i];
      }
   }
   // BGFLOAT **utilizationHistory_;
   {
      BGFLOAT *cpuUtilizationHistory[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuUtilizationHistory, allVertices.utilizationHistory_,
                              numberOfVertices * sizeof(BGFLOAT *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaMemcpy(utilizationHistory_[i].data(), cpuUtilizationHistory[i],
                                 totalTimeSteps * sizeof(BGFLOAT), cudaMemcpyDeviceToHost));
      }
   }
   // int *utilizationHistoryBufferFront_;
   {
      int cpuQueueFront[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuQueueFront, allVertices.utilizationHistoryBufferFront_, numberOfVertices * sizeof(int),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         utilizationHistory_[i].bufferFront_ = cpuQueueFront[i];
      }
   }
   // int *utilizationHistoryBufferEnd_;
   {
      int cpuQueueEnd[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuQueueEnd, allVertices.utilizationHistoryBufferEnd_, numberOfVertices * sizeof(int),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         utilizationHistory_[i].bufferEnd_ = cpuQueueEnd[i];
      }
   }
   // int *utilizationHistoryEpochStart_;
   {
      int cpuEpochStart[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuEpochStart, allVertices.utilizationHistoryEpochStart_, numberOfVertices * sizeof(int),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         utilizationHistory_[i].epochStart_ = cpuEpochStart[i];
      }
   }
   // int *utilizationHistoryNumElementsInEpoch_;
   {
      int cpuElementsInEpoch[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuElementsInEpoch, allVertices.utilizationHistoryNumElementsInEpoch_, numberOfVertices * sizeof(int),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         utilizationHistory_[i].numElementsInEpoch_ = cpuElementsInEpoch[i];
      }
   }
   // int **vertexQueuesBufferVertexId_;
   // uint64_t **vertexQueuesBufferTime_;
   // int **vertexQueuesBufferDuration_;
   // BGFLOAT **vertexQueuesBufferX_;
   // BGFLOAT **vertexQueuesBufferY_;
   // int **vertexQueuesBufferPatience_;
   // int **vertexQueuesBufferOnSiteTime_;
   // int **vertexQueuesBufferResponderType_;
   // uint64_t *vertexQueuesFront_;
   // uint64_t *vertexQueuesEnd_;
   copyVertexQueuesFromDevice(numberOfVertices, stepsPerEpoch, allVertices);
   // int *droppedCalls_;
   HANDLE_ERROR(cudaMemcpy(droppedCalls_.data(), allVertices.droppedCalls_, numberOfVertices * sizeof(int),
                           cudaMemcpyDeviceToHost));
   // int *receivedCalls_;
   HANDLE_ERROR(cudaMemcpy(receivedCalls_.data(), allVertices.receivedCalls_, numberOfVertices * sizeof(int),
                           cudaMemcpyDeviceToHost));
   // int *busyServers_;
   HANDLE_ERROR(cudaMemcpy(busyServers_.data(), allVertices.busyServers_, numberOfVertices * sizeof(int),
                           cudaMemcpyDeviceToHost));
   // int *numServers_;
   HANDLE_ERROR(cudaMemcpy(numServers_.data(), allVertices.numServers_, numberOfVertices * sizeof(int),
                           cudaMemcpyDeviceToHost));
   // int *numTrunks_;
   HANDLE_ERROR(cudaMemcpy(numTrunks_.data(), allVertices.numTrunks_, numberOfVertices * sizeof(int),
                           cudaMemcpyDeviceToHost));
   // int **servingCallBufferVertexId_;
   // uint64_t **servingCallBufferTime_;
   // int **servingCallBufferDuration_;
   // BGFLOAT **servingCallBufferX_;
   // BGFLOAT **servingCallBufferY_;
   // int **servingCallBufferPatience_;
   // int **servingCallBufferOnSiteTime_;
   // int **servingCallBufferResponderType_;
   copyServingCallFromDevice(numberOfVertices, allVertices);
   // uint64_t **answerTime_;
   {
      uint64_t *cpuAnswerTime[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuAnswerTime, allVertices.answerTime_,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaMemcpy(answerTime_[i].data(), cpuAnswerTime[i],
                                 maxNumberOfServers_ * sizeof(uint64_t), cudaMemcpyDeviceToHost));
      }
   }
   // int **serverCountdown_;
   {
      int *cpuServerCountdown[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(cpuServerCountdown, allVertices.serverCountdown_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaMemcpy(serverCountdown_[i].data(), cpuServerCountdown[i],
                                 maxNumberOfServers_ * sizeof(int), cudaMemcpyDeviceToHost));
      }
   }
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
   Simulator &simulator = Simulator::getInstance();
   const int threadsPerBlock = 256;
   int blocksPerGrid
      = (simulator.getTotalVertices() + threadsPerBlock - 1) / threadsPerBlock;
   int totalNumberOfEvents = inputManager_.getTotalNumberOfEvents();
   uint64_t stepsPerEpoch = simulator.getEpochDuration() / simulator.getDeltaT();
   uint64_t totalTimeSteps = stepsPerEpoch * simulator.getNumEpochs();
   Layout &layout = simulator.getModel().getLayout();
   Layout911 &layout911 = dynamic_cast<Layout911 &>(layout);
   BGFLOAT *xLoc_device = layout911.xloc_.getDevicePointer();
   BGFLOAT *yLoc_device = layout911.yloc_.getDevicePointer();
   LOG4CPLUS_DEBUG(vertexLogger_, "blocksPerGrid: " << blocksPerGrid << " threadsPerBlock: " << threadsPerBlock);
   // Advance vertices ------------->
   advance911VerticesDevice<<<blocksPerGrid, threadsPerBlock>>>(size_,
                                                                totalNumberOfEvents,
                                                                stepsPerEpoch,
                                                                totalTimeSteps,
                                                                g_simulationStep,
                                                                avgDrivingSpeed_,
                                                                pi,
                                                                redialP_,
                                                                xLoc_device,
                                                                yLoc_device,
                                                                (All911VerticesDeviceProperties *)allVerticesDevice, 
                                                                (All911EdgesDeviceProperties *)allEdgesDevice, 
                                                                edgeIndexMapDevice);
   cudaDeviceSynchronize();
}

__global__ void advance911VerticesDevice(int totalVertices,
                                         int totalNumberOfEvents,
                                         uint64_t stepsPerEpoch,
                                         uint64_t totalTimeSteps,
                                         uint64_t simulationStep,
                                         BGFLOAT drivingSpeed,
                                         BGFLOAT pi,
                                         BGFLOAT redialProbability,
                                         BGFLOAT *xLocation,
                                         BGFLOAT *yLocation,
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
      case 3: //CALR
         advanceCALRVerticesDevice(idx, totalNumberOfEvents, stepsPerEpoch, simulationStep, 1.0f, redialProbability, allVerticesDevice, allEdgesDevice, edgeIndexMapDevice);
         break;
      case 4: //PSAP
         advancePSAPVerticesDevice(idx, totalNumberOfEvents, stepsPerEpoch, totalTimeSteps, simulationStep, xLocation, yLocation, allVerticesDevice, allEdgesDevice, edgeIndexMapDevice);
         break;
      case 5: //EMS
      case 6: //FIRE
      case 7: //LAW
         advanceRESPVerticesDevice(idx, totalNumberOfEvents, stepsPerEpoch, totalTimeSteps, simulationStep, drivingSpeed, pi, xLocation, yLocation, allVerticesDevice, allEdgesDevice, edgeIndexMapDevice);
         break;
      default:
         printf("ERROR: Vertex is of unknown type [%d]\n", allVerticesDevice->vertexType_[idx]);
   }
}

///  CUDA code for advancing Caller region vertices
///
__device__ void advanceCALRVerticesDevice(int vertexId,
                                             int totalNumberOfEvents,
                                             uint64_t stepsPerEpoch,
                                             uint64_t simulationStep,
                                             BGFLOAT redialValue,
                                             BGFLOAT redialProbability,
                                             All911VerticesDeviceProperties *allVerticesDevice, 
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
      if (!allEdgesDevice->isRedial_[edgeIdx] && redialValue >= redialProbability) {
         // We only make the edge available if no readialing occurs.
         allEdgesDevice->isAvailable_[edgeIdx] = true;
         //LOG4CPLUS_DEBUG(vertexLogger_, "Did not redial at time: " << edges911.call_[edgeIdx].time);
      } else {
         // Keep the edge unavailable but mark it as a redial
         allEdgesDevice->isRedial_[edgeIdx] = true;
      }
   }

   // peek at the next call in the queue
   uint64_t &queueEndIndex = allVerticesDevice->vertexQueuesEnd_[vertexId];
   if (allEdgesDevice->isAvailable_[edgeIdx] && 
      (allVerticesDevice->vertexQueuesFront_[vertexId] != queueEndIndex) && 
      allVerticesDevice->vertexQueuesBufferTime_[vertexId][queueEndIndex] <= simulationStep) {
      // Place new call in the edge going to the PSAP
      if (!allEdgesDevice->isAvailable_[edgeIdx]) {
         printf("ERROR: Edge ID [%d] already has a call for vertex ID [%d]\n", edgeIdx, vertexId);
         return;
      }
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
      queueEndIndex = (queueEndIndex + 1) % stepsPerEpoch;
      allEdgesDevice->isAvailable_[edgeIdx] = false;
   }
}

///  CUDA code for advancing PSAP vertices
///
__device__ void advancePSAPVerticesDevice(int vertexIdx,
                                             int totalNumberOfEvents,
                                             uint64_t stepsPerEpoch,
                                             uint64_t totalTimeSteps,
                                             uint64_t simulationStep,
                                             BGFLOAT *xLocation,
                                             BGFLOAT *yLocation,
                                             All911VerticesDeviceProperties *allVerticesDevice, 
                                             All911EdgesDeviceProperties *allEdgesDevice, 
                                             EdgeIndexMapDevice *edgeIndexMapDevice)
{
   int numberOfServers = allVerticesDevice->numServers_[vertexIdx];
   // Loop over all servers and free the ones finishing serving calls
   int numberOfAvailableServers = 0;
   unsigned char* availableServers = (unsigned char*) malloc(numberOfServers * sizeof(unsigned char));
   // Sanity check that malloc was successful
   if (availableServers == nullptr) {
      printf("ERROR: Failed to allocate memory for availableServers used by vertex ID [%d]\n", vertexIdx);
      return;
   }
   // Initialize to no servers having been assigned a call yet
   for (BGSIZE serverIndex = 0; serverIndex < numberOfServers; serverIndex++) {
      availableServers[serverIndex] = false;
   }
   for (size_t server = 0; server < numberOfServers; ++server) {
      if (allVerticesDevice->serverCountdown_[vertexIdx][server] == 0) {
         // Server is available to take calls. This check is needed because calls
         // could have duration of zero or server has not been assigned a call yet
         availableServers[server] = true;
         numberOfAvailableServers++;
      } else if (--allVerticesDevice->serverCountdown_[vertexIdx][server] == 0) {
         // Server becomes free to take calls
         // TODO: What about wrap-up time?

         //Store call metrics
         // Store wasAbandonedHistory
         // EventBuffer::insertEvent
         if (allVerticesDevice->wasAbandonedHistoryNumElementsInEpoch_[vertexIdx] >= totalNumberOfEvents) {
            printf("ERROR: wasAbandonHistory buffer is full. Vertex ID [%d] Buffer size [%d] Number of Elements in Epoch [%d]\n", vertexIdx, totalNumberOfEvents, allVerticesDevice->wasAbandonedHistoryNumElementsInEpoch_[vertexIdx]);
            return;
         }
         int &abandonedHistoryQueueEnd = allVerticesDevice->wasAbandonedHistoryBufferEnd_[vertexIdx];
         allVerticesDevice->wasAbandonedHistory_[vertexIdx][abandonedHistoryQueueEnd] = false;
         abandonedHistoryQueueEnd = (abandonedHistoryQueueEnd + 1) % totalNumberOfEvents;
         allVerticesDevice->wasAbandonedHistoryNumElementsInEpoch_[vertexIdx]++;
         // Store beginTimeHistory
         // EventBuffer::insertEvent
         if (allVerticesDevice->beginTimeHistoryNumElementsInEpoch_[vertexIdx] >= totalNumberOfEvents) {
            printf("ERROR: beginTimeHistory buffer is full. Vertex ID [%d] Buffer size [%d] Number of Elements in Epoch [%d]\n", vertexIdx, totalNumberOfEvents, allVerticesDevice->beginTimeHistoryNumElementsInEpoch_[vertexIdx]);
            return;
         }
         int &beginHistoryQueueEnd = allVerticesDevice->beginTimeHistoryBufferEnd_[vertexIdx];
         allVerticesDevice->beginTimeHistory_[vertexIdx][beginHistoryQueueEnd] = allVerticesDevice->servingCallBufferTime_[vertexIdx][server];
         beginHistoryQueueEnd = (beginHistoryQueueEnd + 1) % totalNumberOfEvents;
         allVerticesDevice->beginTimeHistoryNumElementsInEpoch_[vertexIdx]++;
         // Store answerTimeHistory
         // EventBuffer::insertEvent
         if (allVerticesDevice->answerTimeHistoryNumElementsInEpoch_[vertexIdx] >= totalNumberOfEvents) {
            printf("ERROR: answerTimeHistory buffer is full. Vertex ID [%d] Buffer size [%d] Number of Elements in Epoch [%d]\n", vertexIdx, totalNumberOfEvents, allVerticesDevice->answerTimeHistoryNumElementsInEpoch_[vertexIdx]);
            return;
         }
         int &answerHistoryQueueEnd = allVerticesDevice->answerTimeHistoryBufferEnd_[vertexIdx];
         allVerticesDevice->answerTimeHistory_[vertexIdx][answerHistoryQueueEnd] = allVerticesDevice->answerTime_[vertexIdx][server];
         answerHistoryQueueEnd = (answerHistoryQueueEnd + 1) % totalNumberOfEvents;
         allVerticesDevice->answerTimeHistoryNumElementsInEpoch_[vertexIdx]++;
         // Store endTimeHistory
         // EventBuffer::insertEvent
         if (allVerticesDevice->endTimeHistoryNumElementsInEpoch_[vertexIdx] >= totalNumberOfEvents) {
            printf("ERROR: endTimeHistory buffer is full. Vertex ID [%d] Buffer size [%d] Number of Elements in Epoch [%d]\n", vertexIdx, totalNumberOfEvents, allVerticesDevice->endTimeHistoryNumElementsInEpoch_[vertexIdx]);
            return;
         }
         int &endHistoryQueueEnd = allVerticesDevice->endTimeHistoryBufferEnd_[vertexIdx];
         allVerticesDevice->endTimeHistory_[vertexIdx][endHistoryQueueEnd] = simulationStep;
         endHistoryQueueEnd = (endHistoryQueueEnd + 1) % totalNumberOfEvents;
         allVerticesDevice->endTimeHistoryNumElementsInEpoch_[vertexIdx]++;

         int requiredType = allVerticesDevice->servingCallBufferResponderType_[vertexIdx][server];

         // loop over the outgoing edges looking for the responder with the shortest
         // Euclidean distance to the call's location.
         BGSIZE startOutEdg = edgeIndexMapDevice->outgoingEdgeBegin_[vertexIdx];
         BGSIZE outEdgCount = edgeIndexMapDevice->outgoingEdgeCount_[vertexIdx];

         BGSIZE resp, respEdge;
         BGFLOAT minDistance = FLT_MAX;
         for (BGSIZE eIdxMap = startOutEdg; eIdxMap < startOutEdg + outEdgCount; ++eIdxMap) {
            BGSIZE outEdg = edgeIndexMapDevice->outgoingEdgeIndexMap_[eIdxMap];
            if (!allEdgesDevice->inUse_[outEdg]) {
               printf("ERROR: Edge must be in use. Edge ID [%d] Vertex ID [%d]\n", outEdg, vertexIdx);
               return;
            }

            BGSIZE dstVertex = allEdgesDevice->destVertexIndex_[outEdg];
            if (allVerticesDevice->vertexType_[dstVertex] == requiredType) {
               //  call x
               BGFLOAT callX = allVerticesDevice->servingCallBufferX_[vertexIdx][server];
               //  call y
               BGFLOAT callY = allVerticesDevice->servingCallBufferY_[vertexIdx][server];
               //  Vertex x
               BGFLOAT dstVertexLocationX = xLocation[dstVertex];
               //  Vertex y
               BGFLOAT dstVertexLocationY = yLocation[dstVertex];
               // Calculates the distance between the given vertex and the (x, y) coordinates of a call
               BGFLOAT distance = sqrtf(powf(callX - dstVertexLocationX, 2) + (powf(callY - dstVertexLocationY, 2)));

               if (distance < minDistance) {
                  minDistance = distance;
                  resp = dstVertex;
                  respEdge = outEdg;
               }
            }
         }

         // We must have found the closest responder of the right type
         if (minDistance >= FLT_MAX) {
            printf("ERROR: Distance found was not the minimum distance. Distance [%f] Responder Edge ID [%u] Vertex ID [%d]\n", minDistance, respEdge, vertexIdx);
            return;
         }
         if (allVerticesDevice->vertexType_[resp] != requiredType) {
            printf("ERROR: Responder vertex was the wrong type. Responder Type [%d] Required Type [%d]\n", allVerticesDevice->vertexType_[respEdge], requiredType);
            return;
         }

         //int responder = allEdgesDevice->destVertexIndex_[respEdge];
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
         availableServers[server] = true;
         numberOfAvailableServers++;
      }
   }

   // Assign calls to servers until either no servers are available or
   // there are no more calls in the waiting queue
   int currentlyAvailableServers = numberOfAvailableServers;
   uint64_t queueFrontIndex = allVerticesDevice->vertexQueuesFront_[vertexIdx];
   uint64_t &queueEndIndex = allVerticesDevice->vertexQueuesEnd_[vertexIdx];
   while (currentlyAvailableServers > 0 && !(queueFrontIndex == queueEndIndex)) {
      // TODO: calls with duration of zero are being added but because countdown will be zero
      //       they don't show up in the logs
      int callId = allVerticesDevice->vertexQueuesBufferVertexId_[vertexIdx][queueEndIndex];
      uint64_t callTime = allVerticesDevice->vertexQueuesBufferTime_[vertexIdx][queueEndIndex];
      int callDuration = allVerticesDevice->vertexQueuesBufferDuration_[vertexIdx][queueEndIndex];
      BGFLOAT callX = allVerticesDevice->vertexQueuesBufferX_[vertexIdx][queueEndIndex];
      BGFLOAT callY = allVerticesDevice->vertexQueuesBufferY_[vertexIdx][queueEndIndex];
      int callPatience = allVerticesDevice->vertexQueuesBufferPatience_[vertexIdx][queueEndIndex];
      int callOnSiteTime = allVerticesDevice->vertexQueuesBufferOnSiteTime_[vertexIdx][queueEndIndex];
      int callResponderType = allVerticesDevice->vertexQueuesBufferResponderType_[vertexIdx][queueEndIndex];
      queueEndIndex = (queueEndIndex + 1) % stepsPerEpoch;

      if (callPatience < (simulationStep - callTime)) {
         // If the patience time is less than the waiting time, the call is abandoned
         // Store wasAbandonedHistory
         // EventBuffer::insertEvent
         if (allVerticesDevice->wasAbandonedHistoryNumElementsInEpoch_[vertexIdx] >= totalNumberOfEvents) {
            printf("ERROR: wasAbandonHistory buffer is full. Vertex ID [%d] Buffer size [%d] Number of Elements in Epoch [%d]\n", vertexIdx, totalNumberOfEvents, allVerticesDevice->wasAbandonedHistoryNumElementsInEpoch_[vertexIdx]);
            return;
         }
         int &abandonedHistoryQueueEnd = allVerticesDevice->wasAbandonedHistoryBufferEnd_[vertexIdx];
         allVerticesDevice->wasAbandonedHistory_[vertexIdx][abandonedHistoryQueueEnd] = true;
         abandonedHistoryQueueEnd = (abandonedHistoryQueueEnd + 1) % totalNumberOfEvents;
         allVerticesDevice->wasAbandonedHistoryNumElementsInEpoch_[vertexIdx]++;
         // Store beginTimeHistory
         // EventBuffer::insertEvent
         if (allVerticesDevice->beginTimeHistoryNumElementsInEpoch_[vertexIdx] >= totalNumberOfEvents) {
            printf("ERROR: beginTimeHistory buffer is full. Vertex ID [%d] Buffer size [%d] Number of Elements in Epoch [%d]\n", vertexIdx, totalNumberOfEvents, allVerticesDevice->beginTimeHistoryNumElementsInEpoch_[vertexIdx]);
            return;
         }
         int &beginHistoryQueueEnd = allVerticesDevice->beginTimeHistoryBufferEnd_[vertexIdx];
         allVerticesDevice->beginTimeHistory_[vertexIdx][beginHistoryQueueEnd] = callTime;
         beginHistoryQueueEnd = (beginHistoryQueueEnd + 1) % totalNumberOfEvents;
         allVerticesDevice->beginTimeHistoryNumElementsInEpoch_[vertexIdx]++;
         // Answer time and end time get zero as sentinel for non-valid values
         // Store answerTimeHistory
         // EventBuffer::insertEvent
         if (allVerticesDevice->answerTimeHistoryNumElementsInEpoch_[vertexIdx] >= totalNumberOfEvents) {
            printf("ERROR: answerTimeHistory buffer is full. Vertex ID [%d] Buffer size [%d] Number of Elements in Epoch [%d]\n", vertexIdx, totalNumberOfEvents, allVerticesDevice->answerTimeHistoryNumElementsInEpoch_[vertexIdx]);
            return;
         }
         int &answerHistoryQueueEnd = allVerticesDevice->answerTimeHistoryBufferEnd_[vertexIdx];
         allVerticesDevice->answerTimeHistory_[vertexIdx][answerHistoryQueueEnd] = 0;
         answerHistoryQueueEnd = (answerHistoryQueueEnd + 1) % totalNumberOfEvents;
         allVerticesDevice->answerTimeHistoryNumElementsInEpoch_[vertexIdx]++;
         // Store endTimeHistory
         // EventBuffer::insertEvent
         if (allVerticesDevice->endTimeHistoryNumElementsInEpoch_[vertexIdx] >= totalNumberOfEvents) {
            printf("ERROR: endTimeHistory buffer is full. Vertex ID [%d] Buffer size [%d] Number of Elements in Epoch [%d]\n", vertexIdx, totalNumberOfEvents, allVerticesDevice->endTimeHistoryNumElementsInEpoch_[vertexIdx]);
            return;
         }
         int &endHistoryQueueEnd = allVerticesDevice->endTimeHistoryBufferEnd_[vertexIdx];
         allVerticesDevice->endTimeHistory_[vertexIdx][endHistoryQueueEnd] = 0;
         endHistoryQueueEnd = (endHistoryQueueEnd + 1) % totalNumberOfEvents;
         allVerticesDevice->endTimeHistoryNumElementsInEpoch_[vertexIdx]++;
      } else {
         // The available server starts serving the call
         int availServer;
         for(BGSIZE serverIndex = 0; serverIndex < numberOfServers; serverIndex++) {
            if (availableServers[serverIndex] == true) {
               // If server is available, have that server serve the call
               availServer = serverIndex;
               availableServers[serverIndex] = false;
               currentlyAvailableServers--;
               break;
            }
         }
         allVerticesDevice->servingCallBufferVertexId_[vertexIdx][availServer] = callId;
         allVerticesDevice->servingCallBufferTime_[vertexIdx][availServer] = callTime;
         allVerticesDevice->servingCallBufferDuration_[vertexIdx][availServer] = callDuration;
         allVerticesDevice->servingCallBufferX_[vertexIdx][availServer] = callX;
         allVerticesDevice->servingCallBufferY_[vertexIdx][availServer] = callY;
         allVerticesDevice->servingCallBufferPatience_[vertexIdx][availServer] = callPatience;
         allVerticesDevice->servingCallBufferOnSiteTime_[vertexIdx][availServer] = callOnSiteTime;
         allVerticesDevice->servingCallBufferResponderType_[vertexIdx][availServer] = callResponderType;

         allVerticesDevice->answerTime_[vertexIdx][availServer] = simulationStep;
         allVerticesDevice->serverCountdown_[vertexIdx][availServer] = callDuration;
      }
   }

   // Update number of busy servers. This is used to check if there is space in the queue
   allVerticesDevice->busyServers_[vertexIdx] = allVerticesDevice->numServers_[vertexIdx] - numberOfAvailableServers;

   // Update queueLength and utilization histories
   // Compute the size of the destination queue for queue length
   uint64_t queueSize;
   queueFrontIndex = allVerticesDevice->vertexQueuesFront_[vertexIdx];
   queueEndIndex = allVerticesDevice->vertexQueuesEnd_[vertexIdx];
   if (queueFrontIndex >= queueEndIndex) {
      queueSize = queueFrontIndex - queueEndIndex;
   } else {
      queueSize = totalTimeSteps + queueFrontIndex - queueEndIndex;
   }
   // EventBuffer::insertEvent
   if (allVerticesDevice->queueLengthHistoryNumElementsInEpoch_[vertexIdx] >= totalTimeSteps) {
      printf("ERROR: queueLengthHistory buffer is full. Vertex ID [%d] Buffer size [%" PRIu64 "] Number of Elements in Epoch [%d]\n", vertexIdx, totalTimeSteps, allVerticesDevice->queueLengthHistoryNumElementsInEpoch_[vertexIdx]);
      return;
   }
   int &queueLengthHistoryQueueEnd = allVerticesDevice->queueLengthHistoryBufferEnd_[vertexIdx];
   allVerticesDevice->queueLengthHistory_[vertexIdx][queueLengthHistoryQueueEnd] = queueSize;
   queueLengthHistoryQueueEnd = (queueLengthHistoryQueueEnd + 1) % totalTimeSteps;
   allVerticesDevice->queueLengthHistoryNumElementsInEpoch_[vertexIdx]++;
   // EventBuffer::insertEvent
   if (allVerticesDevice->utilizationHistoryNumElementsInEpoch_[vertexIdx] >= totalTimeSteps) {
      printf("ERROR: utilizationHistory buffer is full. Vertex ID [%d] Buffer size [%" PRIu64 "] Number of Elements in Epoch [%d]\n", vertexIdx, totalTimeSteps, allVerticesDevice->utilizationHistoryNumElementsInEpoch_[vertexIdx]);
      return;
   }
   int &utilizationHistoryQueueEnd = allVerticesDevice->utilizationHistoryBufferEnd_[vertexIdx];
   allVerticesDevice->utilizationHistory_[vertexIdx][utilizationHistoryQueueEnd] 
      = static_cast<float>(allVerticesDevice->busyServers_[vertexIdx]) / allVerticesDevice->numServers_[vertexIdx];
   utilizationHistoryQueueEnd = (utilizationHistoryQueueEnd + 1) % totalTimeSteps;
   allVerticesDevice->utilizationHistoryNumElementsInEpoch_[vertexIdx]++;
}

///  CUDA code for advancing emergency responder vertices
///
__device__ void advanceRESPVerticesDevice(int vertexIdx,
                                             int totalNumberOfEvents,
                                             uint64_t stepsPerEpoch,
                                             uint64_t totalTimeSteps,
                                             uint64_t simulationStep,
                                             BGFLOAT drivingSpeed,
                                             BGFLOAT pi, BGFLOAT *xLocation, BGFLOAT *yLocation, All911VerticesDeviceProperties *allVerticesDevice, All911EdgesDeviceProperties *allEdgesDevice, EdgeIndexMapDevice *edgeIndexMapDevice)
{
   int numberOfUnits = allVerticesDevice->numServers_[vertexIdx];
   // Free the units finishing up with emergency responses
   int numberOfAvailableUnits = 0;
   unsigned char* availableUnits = (unsigned char*) malloc(numberOfUnits * sizeof(unsigned char));
   // Sanity check that malloc was successful
   if (availableUnits == nullptr) {
      printf("ERROR: Failed to allocate memory for availableUnits used by vertex ID [%d]\n", vertexIdx);
      return;
   }
   for (BGSIZE unitIndex = 0; unitIndex < numberOfUnits; unitIndex++) {
      availableUnits[unitIndex] = false;
   }
   for (size_t unit = 0; unit < numberOfUnits; ++unit) {
      if (allVerticesDevice->serverCountdown_[vertexIdx][unit] == 0) {
         // Unit is available
         availableUnits[unit] = true;
         numberOfAvailableUnits++;
      } else if (--allVerticesDevice->serverCountdown_[vertexIdx][unit] == 0) {
         // Unit becomes available to responde to new incidents

         //Store incident response metrics
         // Store wasAbandonedHistory
         // EventBuffer::insertEvent
         if (allVerticesDevice->wasAbandonedHistoryNumElementsInEpoch_[vertexIdx] >= totalNumberOfEvents) {
            printf("ERROR: wasAbandonHistory buffer is full. Vertex ID [%d] Buffer size [%d] Number of Elements in Epoch [%d]\n", vertexIdx, totalNumberOfEvents, allVerticesDevice->wasAbandonedHistoryNumElementsInEpoch_[vertexIdx]);
            return;
         }
         int &abandonedHistoryQueueEnd = allVerticesDevice->wasAbandonedHistoryBufferEnd_[vertexIdx];
         allVerticesDevice->wasAbandonedHistory_[vertexIdx][abandonedHistoryQueueEnd] = false;
         abandonedHistoryQueueEnd = (abandonedHistoryQueueEnd + 1) % totalNumberOfEvents;
         allVerticesDevice->wasAbandonedHistoryNumElementsInEpoch_[vertexIdx]++;
         // Store beginTimeHistory
         // EventBuffer::insertEvent
         if (allVerticesDevice->beginTimeHistoryNumElementsInEpoch_[vertexIdx] >= totalNumberOfEvents) {
            printf("ERROR: beginTimeHistory buffer is full. Vertex ID [%d] Buffer size [%d] Number of Elements in Epoch [%d]\n", vertexIdx, totalNumberOfEvents, allVerticesDevice->beginTimeHistoryNumElementsInEpoch_[vertexIdx]);
            return;
         }
         int &beginHistoryQueueEnd = allVerticesDevice->beginTimeHistoryBufferEnd_[vertexIdx];
         allVerticesDevice->beginTimeHistory_[vertexIdx][beginHistoryQueueEnd] = allVerticesDevice->servingCallBufferTime_[vertexIdx][unit];
         beginHistoryQueueEnd = (beginHistoryQueueEnd + 1) % totalNumberOfEvents;
         allVerticesDevice->beginTimeHistoryNumElementsInEpoch_[vertexIdx]++;
         // Store answerTimeHistory
         // EventBuffer::insertEvent
         if (allVerticesDevice->answerTimeHistoryNumElementsInEpoch_[vertexIdx] >= totalNumberOfEvents) {
            printf("ERROR: answerTimeHistory buffer is full. Vertex ID [%d] Buffer size [%d] Number of Elements in Epoch [%d]\n", vertexIdx, totalNumberOfEvents, allVerticesDevice->answerTimeHistoryNumElementsInEpoch_[vertexIdx]);
            return;
         }
         int &answerHistoryQueueEnd = allVerticesDevice->answerTimeHistoryBufferEnd_[vertexIdx];
         allVerticesDevice->answerTimeHistory_[vertexIdx][answerHistoryQueueEnd] = allVerticesDevice->answerTime_[vertexIdx][unit];
         answerHistoryQueueEnd = (answerHistoryQueueEnd + 1) % totalNumberOfEvents;
         allVerticesDevice->answerTimeHistoryNumElementsInEpoch_[vertexIdx]++;
         // Store endTimeHistory
         // EventBuffer::insertEvent
         if (allVerticesDevice->endTimeHistoryNumElementsInEpoch_[vertexIdx] >= totalNumberOfEvents) {
            printf("ERROR: endTimeHistory buffer is full. Vertex ID [%d] Buffer size [%d] Number of Elements in Epoch [%d]\n", vertexIdx, totalNumberOfEvents, allVerticesDevice->endTimeHistoryNumElementsInEpoch_[vertexIdx]);
            return;
         }
         int &endHistoryQueueEnd = allVerticesDevice->endTimeHistoryBufferEnd_[vertexIdx];
         allVerticesDevice->endTimeHistory_[vertexIdx][endHistoryQueueEnd] = simulationStep;
         endHistoryQueueEnd = (endHistoryQueueEnd + 1) % totalNumberOfEvents;
         allVerticesDevice->endTimeHistoryNumElementsInEpoch_[vertexIdx]++;

         // Unit is added to available units
         availableUnits[unit] = true;
         numberOfAvailableUnits++;
      }
   }


   // Assign reponse dispatches until no units are available or there are no more
   // incidents in the waiting queue
   uint64_t queueFrontIndex = allVerticesDevice->vertexQueuesFront_[vertexIdx];
   uint64_t &queueEndIndex = allVerticesDevice->vertexQueuesEnd_[vertexIdx];
   for (size_t unit = 0; unit < numberOfAvailableUnits && !(queueFrontIndex == queueEndIndex);
        ++unit) {
      int incidentId = allVerticesDevice->vertexQueuesBufferVertexId_[vertexIdx][queueEndIndex];
      uint64_t incidentTime = allVerticesDevice->vertexQueuesBufferTime_[vertexIdx][queueEndIndex];
      int incidentDuration = allVerticesDevice->vertexQueuesBufferDuration_[vertexIdx][queueEndIndex];
      BGFLOAT incidentX = allVerticesDevice->vertexQueuesBufferX_[vertexIdx][queueEndIndex];
      BGFLOAT incidentY = allVerticesDevice->vertexQueuesBufferY_[vertexIdx][queueEndIndex];
      int incidentPatience = allVerticesDevice->vertexQueuesBufferPatience_[vertexIdx][queueEndIndex];
      int incidentOnSiteTime = allVerticesDevice->vertexQueuesBufferOnSiteTime_[vertexIdx][queueEndIndex];
      int incidentResponderType = allVerticesDevice->vertexQueuesBufferResponderType_[vertexIdx][queueEndIndex];
      queueEndIndex = (queueEndIndex + 1) % stepsPerEpoch;

      // The available unit starts serving the call
      int availUnit;
      for(BGSIZE unitIndex = 0; unitIndex < numberOfUnits; unitIndex++) {
         if (availableUnits[unitIndex] == true) {
            // If server is available, have that server serve the call
            availUnit = unitIndex;
            availableUnits[unitIndex] = false;
            break;
         }
      }
      allVerticesDevice->servingCallBufferVertexId_[vertexIdx][availUnit] = incidentId;
      allVerticesDevice->servingCallBufferTime_[vertexIdx][availUnit] = incidentTime;
      allVerticesDevice->servingCallBufferDuration_[vertexIdx][availUnit] = incidentDuration;
      allVerticesDevice->servingCallBufferX_[vertexIdx][availUnit] = incidentX;
      allVerticesDevice->servingCallBufferY_[vertexIdx][availUnit] = incidentY;
      allVerticesDevice->servingCallBufferPatience_[vertexIdx][availUnit] = incidentPatience;
      allVerticesDevice->servingCallBufferOnSiteTime_[vertexIdx][availUnit] = incidentOnSiteTime;
      allVerticesDevice->servingCallBufferResponderType_[vertexIdx][availUnit] = incidentResponderType;

      allVerticesDevice->answerTime_[vertexIdx][availUnit] = simulationStep;

      // We need to calculate the distance in miles but the x and y coordinates
      // represent, respectively, degrees of longitude and latitude.
      // One degree of latitude is aproximately 69 miles regardles of the location. However,
      // a degree of longitude varies, being 69.172 miles at the equator and gradually shrinking
      // to zero at the poles.
      // One degree of longitude can be converted to miles using the following formula:
      //    1 degree of longitude = cos(latitude) * 69.172
      BGFLOAT lngDegreeLength = cos(yLocation[vertexIdx] * (pi / 180)) * 69.172;
      BGFLOAT latDegreeLength = 69.0;
      BGFLOAT deltaLng = incidentX - xLocation[vertexIdx];
      BGFLOAT deltaLat = incidentY - yLocation[vertexIdx];
      BGFLOAT dist2incident
         = sqrtf(powf(deltaLng * lngDegreeLength, 2) + powf(deltaLat * latDegreeLength, 2));

      // Calculate the driving time to the incident in seconds
      BGFLOAT driveTime = (dist2incident / drivingSpeed) * 3600;
      allVerticesDevice->serverCountdown_[vertexIdx][availUnit] = driveTime + incidentOnSiteTime;

      /// Wouldn't this just immediately overwrite the above? Why is it needed?
      allVerticesDevice->serverCountdown_[vertexIdx][availUnit] = incidentDuration;
   }

   // Update number of busy servers. This is used to check if there is space in the queue
   allVerticesDevice->busyServers_[vertexIdx] = numberOfUnits - numberOfAvailableUnits;

   // Update queueLength and utilization histories
   // Compute the size of the destination queue for queue length
   uint64_t queueSize;
   queueFrontIndex = allVerticesDevice->vertexQueuesFront_[vertexIdx];
   queueEndIndex = allVerticesDevice->vertexQueuesEnd_[vertexIdx];
   if (queueFrontIndex >= queueEndIndex) {
      queueSize = queueFrontIndex - queueEndIndex;
   } else {
      queueSize = totalTimeSteps + queueFrontIndex - queueEndIndex;
   }
   // EventBuffer::insertEvent
   if (allVerticesDevice->queueLengthHistoryNumElementsInEpoch_[vertexIdx] >= totalTimeSteps) {
      printf("ERROR: queueLengthHistory buffer is full. Vertex ID [%d] Buffer size [%" PRIu64 "] Number of Elements in Epoch [%d]\n", vertexIdx, totalTimeSteps, allVerticesDevice->queueLengthHistoryNumElementsInEpoch_[vertexIdx]);
      return;
   }
   int &queueLengthHistoryQueueEnd = allVerticesDevice->queueLengthHistoryBufferEnd_[vertexIdx];
   allVerticesDevice->queueLengthHistory_[vertexIdx][queueLengthHistoryQueueEnd] = queueSize;
   queueLengthHistoryQueueEnd = (queueLengthHistoryQueueEnd + 1) % totalTimeSteps;
   allVerticesDevice->queueLengthHistoryNumElementsInEpoch_[vertexIdx]++;
   // EventBuffer::insertEvent
   if (allVerticesDevice->utilizationHistoryNumElementsInEpoch_[vertexIdx] >= totalTimeSteps) {
      printf("ERROR: utilizationHistory buffer is full. Vertex ID [%d] Buffer size [%" PRIu64 "] Number of Elements in Epoch [%d]\n", vertexIdx, totalTimeSteps, allVerticesDevice->utilizationHistoryNumElementsInEpoch_[vertexIdx]);
      return;
   }
   int &utilizationHistoryQueueEnd = allVerticesDevice->utilizationHistoryBufferEnd_[vertexIdx];
   allVerticesDevice->utilizationHistory_[vertexIdx][utilizationHistoryQueueEnd] 
      = static_cast<float>(allVerticesDevice->busyServers_[vertexIdx]) / numberOfUnits;
   utilizationHistoryQueueEnd = (utilizationHistoryQueueEnd + 1) % totalTimeSteps;
   allVerticesDevice->utilizationHistoryNumElementsInEpoch_[vertexIdx]++;
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
   Simulator &simulator = Simulator::getInstance();
   int totalVertices = simulator.getTotalVertices();
   const int threadsPerBlock = 256;
   int blocksPerGrid
      = (totalVertices + threadsPerBlock - 1) / threadsPerBlock;
   uint64_t stepsPerEpoch = simulator.getEpochDuration() / simulator.getDeltaT();
   
   maybeTakeCallFromEdge<<<blocksPerGrid, threadsPerBlock>>>(totalVertices,
                                                             stepsPerEpoch, 
                                                             (All911VerticesDeviceProperties *)allVerticesDevice, 
                                                             (All911EdgesDeviceProperties *)allEdgesDevice,
                                                             edgeIndexMapDevice);
   cudaDeviceSynchronize();
}

__global__ void maybeTakeCallFromEdge(int totalVertices,
                                      uint64_t stepsPerEpoch,
                                      All911VerticesDeviceProperties *allVerticesDevice,
                                      All911EdgesDeviceProperties *allEdgesDevice,
                                      EdgeIndexMapDevice *edgeIndexMapDevice)
{
   // The usual thread ID calculation and guard against excess threads
   // (beyond the number of vertices, in this case).
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx >= totalVertices)
      return;
   
   // TODO911: Caller Regions will have different behaviour
   if (allVerticesDevice->vertexType_[idx] == 3) {
      return;
   }
   
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
      if (dstIndex != idx) {
         printf("ERROR: The destination vertex is responsible for pulling in it's calls. Destination Vertex ID [%d] Vertex ID [%d]\n", dstIndex, idx);
         return;
      }

      // Compute the size of the destination queue
      uint64_t dstQueueSize;
      uint64_t queueFrontIndex = allVerticesDevice->vertexQueuesFront_[dstIndex];
      uint64_t queueEndIndex = allVerticesDevice->vertexQueuesEnd_[dstIndex];
      if (queueFrontIndex >= queueEndIndex) {
         dstQueueSize = queueFrontIndex - queueEndIndex;
      } else {
         dstQueueSize = stepsPerEpoch + queueFrontIndex - queueEndIndex;
      }

      // Compute the capacity of the destination queue
      int dstQueueCapacity = stepsPerEpoch - 1;

      // Get the number fo busy servers at the destination vertex
      int dstBusyServers = allVerticesDevice->busyServers_[dstIndex];

      if (dstQueueSize >= (dstQueueCapacity - dstBusyServers)) {
         // Call is dropped because there is no space in the waiting queue
         if (!allEdgesDevice->isRedial_[edgeIdx]) {
            // Only count the dropped call if it's not a redial
            allVerticesDevice->droppedCalls_[dstIndex]++;
            // Record that we received a call
            allVerticesDevice->receivedCalls_[dstIndex]++;
         }
      } else {
         // Transfer call to destination
         // We throw an error if the buffer is full
         if (((queueFrontIndex + 1) % stepsPerEpoch) == queueEndIndex) {
            printf("ERROR: Vertex queue is full. Vertex ID [%d] Front Index [%" PRIu64 "] End Index [%" PRIu64 "] Buffer size [%" PRIu64 "]\n", dstIndex, queueFrontIndex, queueEndIndex, stepsPerEpoch);
            return;
         }
         // Insert the new element and increment the front index
         allVerticesDevice->vertexQueuesBufferVertexId_[dstIndex][queueFrontIndex] = allEdgesDevice->vertexId_[edgeIdx];
         allVerticesDevice->vertexQueuesBufferTime_[dstIndex][queueFrontIndex] = allEdgesDevice->time_[edgeIdx];
         allVerticesDevice->vertexQueuesBufferDuration_[dstIndex][queueFrontIndex] = allEdgesDevice->duration_[edgeIdx];
         allVerticesDevice->vertexQueuesBufferX_[dstIndex][queueFrontIndex] = allEdgesDevice->x_[edgeIdx];
         allVerticesDevice->vertexQueuesBufferY_[dstIndex][queueFrontIndex] = allEdgesDevice->y_[edgeIdx];
         allVerticesDevice->vertexQueuesBufferPatience_[dstIndex][queueFrontIndex] = allEdgesDevice->patience_[edgeIdx];
         allVerticesDevice->vertexQueuesBufferOnSiteTime_[dstIndex][queueFrontIndex] = allEdgesDevice->onSiteTime_[edgeIdx];
         allVerticesDevice->vertexQueuesBufferResponderType_[dstIndex][queueFrontIndex] = allEdgesDevice->responderType_[edgeIdx];
         allVerticesDevice->vertexQueuesFront_[dstIndex] = (queueFrontIndex + 1) % stepsPerEpoch;
         // Record that we received a call
         allVerticesDevice->receivedCalls_[dstIndex]++;
         allEdgesDevice->isAvailable_[edgeIdx] = true;
         allEdgesDevice->isRedial_[edgeIdx] = false;
      }
   }
}

void All911Vertices::clearVertexHistory(void *allVerticesDevice)
{
   /// What exactly should this clear out? Probably at least the vertex queues
   All911VerticesDeviceProperties allVertices;
   HANDLE_ERROR(cudaMemcpy(&allVertices, allVerticesDevice,
                           sizeof(All911VerticesDeviceProperties), cudaMemcpyDeviceToHost));
   
   int numberOfVertices = Simulator::getInstance().getTotalVertices();
   // uint64_t **beginTimeHistory_
   HANDLE_ERROR(cudaMemset(allVertices.beginTimeHistoryNumElementsInEpoch_, 0, numberOfVertices * sizeof(int)));
   {
      vector<int> epochStart(numberOfVertices);
      for (int i = 0; i < epochStart.size(); ++i) {
         epochStart[i] = beginTimeHistory_[i].bufferEnd_;
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.beginTimeHistoryEpochStart_, epochStart.data(),
                              numberOfVertices * sizeof(int), cudaMemcpyHostToDevice));
   }
   // uint64_t **answerTimeHistory_
   HANDLE_ERROR(cudaMemset(allVertices.answerTimeHistoryNumElementsInEpoch_, 0, numberOfVertices * sizeof(int)));
   {
      vector<int> epochStart(numberOfVertices);
      for (int i = 0; i < epochStart.size(); ++i) {
         epochStart[i] = answerTimeHistory_[i].bufferEnd_;
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.answerTimeHistoryEpochStart_, epochStart.data(),
                              numberOfVertices * sizeof(int), cudaMemcpyHostToDevice));
   }
   // uint64_t **endTimeHistory_
   HANDLE_ERROR(cudaMemset(allVertices.endTimeHistoryNumElementsInEpoch_, 0, numberOfVertices * sizeof(int)));
   {
      vector<int> epochStart(numberOfVertices);
      for (int i = 0; i < epochStart.size(); ++i) {
         epochStart[i] = endTimeHistory_[i].bufferEnd_;
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.endTimeHistoryEpochStart_, epochStart.data(),
                              numberOfVertices * sizeof(int), cudaMemcpyHostToDevice));
   }
   // uint64_t **wasAbandonedHistory_
   HANDLE_ERROR(cudaMemset(allVertices.wasAbandonedHistoryNumElementsInEpoch_, 0, numberOfVertices * sizeof(int)));
   {
      vector<int> epochStart(numberOfVertices);
      for (int i = 0; i < epochStart.size(); ++i) {
         epochStart[i] = wasAbandonedHistory_[i].bufferEnd_;
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.wasAbandonedHistoryEpochStart_, epochStart.data(),
                              numberOfVertices * sizeof(int), cudaMemcpyHostToDevice));
   }
   // uint64_t **queueLengthHistory_
   HANDLE_ERROR(cudaMemset(allVertices.queueLengthHistoryNumElementsInEpoch_, 0, numberOfVertices * sizeof(int)));
   {
      vector<int> epochStart(numberOfVertices);
      for (int i = 0; i < epochStart.size(); ++i) {
         epochStart[i] = queueLengthHistory_[i].bufferEnd_;
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.queueLengthHistoryEpochStart_, epochStart.data(),
                              numberOfVertices * sizeof(int), cudaMemcpyHostToDevice));
   }
   // BGFLOAT **utilizationHistory_
   HANDLE_ERROR(cudaMemset(allVertices.utilizationHistoryNumElementsInEpoch_, 0, numberOfVertices * sizeof(int)));
   {
      vector<int> epochStart(numberOfVertices);
      for (int i = 0; i < epochStart.size(); ++i) {
         epochStart[i] = utilizationHistory_[i].bufferEnd_;
      }
      HANDLE_ERROR(cudaMemcpy(allVertices.utilizationHistoryEpochStart_, epochStart.data(),
                              numberOfVertices * sizeof(int), cudaMemcpyHostToDevice));
   }
}

void All911Vertices::clearVertexQueuesOnDevice(int numberOfVertices, uint64_t stepsPerEpoch, All911VerticesDeviceProperties &allVerticesDevice)
{
   // int **vertexQueuesBufferVertexId_;
   {
      int *callIdCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callIdCpu, allVerticesDevice.vertexQueuesBufferVertexId_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<int> callIdInBuffer;
      // resize to create vector of 0s
      callIdInBuffer.resize(stepsPerEpoch);
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaMemcpy(callIdCpu[i], callIdInBuffer.data(),
                                 stepsPerEpoch * sizeof(int), cudaMemcpyHostToDevice));
      }
   }
   // uint64_t **vertexQueuesBufferTime_;
   {
      uint64_t *callTimeCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callTimeCpu, allVerticesDevice.vertexQueuesBufferTime_,
                              numberOfVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<uint64_t> callTimeInBuffer;
      callTimeInBuffer.resize(stepsPerEpoch);
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaMemcpy(callTimeCpu[i], callTimeInBuffer.data(),
                                 stepsPerEpoch * sizeof(uint64_t), cudaMemcpyHostToDevice));
      }
   }
   // int **vertexQueuesBufferDuration_;
   {
      int *callDurationCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callDurationCpu, allVerticesDevice.vertexQueuesBufferDuration_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<int> callDurationInBuffer;
      callDurationInBuffer.resize(stepsPerEpoch);
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaMemcpy(callDurationCpu[i], callDurationInBuffer.data(),
                                 stepsPerEpoch * sizeof(int), cudaMemcpyHostToDevice));
      }
   }
   // BGFLOAT **vertexQueuesBufferX_;
   {
      BGFLOAT *callLocationXCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callLocationXCpu, allVerticesDevice.vertexQueuesBufferX_,
                              numberOfVertices * sizeof(BGFLOAT *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<BGFLOAT> callLocationXInBuffer;
      callLocationXInBuffer.resize(stepsPerEpoch);
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaMemcpy(callLocationXCpu[i], callLocationXInBuffer.data(),
                                 stepsPerEpoch * sizeof(BGFLOAT), cudaMemcpyHostToDevice));
      }
   }
   // BGFLOAT **vertexQueuesBufferY_;
   {
      BGFLOAT *callLocationYCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callLocationYCpu, allVerticesDevice.vertexQueuesBufferY_,
                              numberOfVertices * sizeof(BGFLOAT *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<BGFLOAT> callLocationYInBuffer;
      callLocationYInBuffer.resize(stepsPerEpoch);
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaMemcpy(callLocationYCpu[i], callLocationYInBuffer.data(),
                                 stepsPerEpoch * sizeof(BGFLOAT), cudaMemcpyHostToDevice));
      }
   }
   // int **vertexQueuesBufferPatience_;
   {
      int *callPatienceCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callPatienceCpu, allVerticesDevice.vertexQueuesBufferPatience_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<int> callPatienceInBuffer;
      callPatienceInBuffer.resize(stepsPerEpoch);
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaMemcpy(callPatienceCpu[i], callPatienceInBuffer.data(),
                                 stepsPerEpoch * sizeof(int), cudaMemcpyHostToDevice));
      }
   }
   // int **vertexQueuesBufferOnSiteTime_;
   {
      int *callOnSiteTimeCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callOnSiteTimeCpu, allVerticesDevice.vertexQueuesBufferOnSiteTime_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<int> callOnSiteTimeInBuffer;
      callOnSiteTimeInBuffer.resize(stepsPerEpoch);
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaMemcpy(callOnSiteTimeCpu[i], callOnSiteTimeInBuffer.data(),
                                 stepsPerEpoch * sizeof(int), cudaMemcpyHostToDevice));
      }
   }
   // int **vertexQueuesBufferResponderType_;
   {
      int *callResponderTypeCpu[numberOfVertices];
      HANDLE_ERROR(cudaMemcpy(callResponderTypeCpu, allVerticesDevice.vertexQueuesBufferResponderType_,
                              numberOfVertices * sizeof(int *), cudaMemcpyDeviceToHost));

      // Using a vector since we are still on the CPU and it's convenient to call data()
      // in memcpy and using the same vector over and over helps with stack memory
      // management
      vector<int> callResponderTypeInBuffer;
      callResponderTypeInBuffer.resize(stepsPerEpoch);
      for (int i = 0; i < numberOfVertices; i++) {
         HANDLE_ERROR(cudaMemcpy(callResponderTypeCpu[i], callResponderTypeInBuffer.data(),
                                 stepsPerEpoch * sizeof(int), cudaMemcpyHostToDevice));
      }
   }
   // uint64_t *vertexQueuesFront_;
   HANDLE_ERROR(cudaMemset(allVerticesDevice.vertexQueuesFront_, 0, numberOfVertices * sizeof(uint64_t)));
   // uint64_t *vertexQueuesEnd_;
   HANDLE_ERROR(cudaMemset(allVerticesDevice.vertexQueuesEnd_, 0, numberOfVertices * sizeof(uint64_t)));
}

/// Copies all inputs scheduled to occur in the upcoming epoch onto device.
void All911Vertices::copyEpochInputsToDevice()
{
   LOG4CPLUS_DEBUG(fileLogger_, "Calling All911Vertices::copyEpochInputsToDevice");
   // The only new inputs are going to be for caller region vertices. However, due to how
   // we have our memory organized on the GPU, we need to copy over all queues instead of
   // just the queues with new inputs.
   All911VerticesDeviceProperties allVertices;
   Simulator &simulator = Simulator::getInstance();
   int numberOfVertices = simulator.getTotalVertices();
   uint64_t stepsPerEpoch = simulator.getEpochDuration() / simulator.getDeltaT();
   GPUModel *gpuModel = static_cast<GPUModel *>(&(simulator.getModel()));
   void *deviceAddress = static_cast<void *>(gpuModel->getAllVerticesDevice());
   HANDLE_ERROR(cudaMemcpy(&allVertices, deviceAddress,
                           sizeof(All911VerticesDeviceProperties), cudaMemcpyDeviceToHost));
   //clearVertexQueuesOnDevice(numberOfVertices, stepsPerEpoch, allVertices);
   copyVertexQueuesToDevice(numberOfVertices, stepsPerEpoch, allVertices);
}

void All911Vertices::setAdvanceVerticesDeviceParams(AllEdges &edges)
{

}