/**
 *  @file All911Edges_d.cpp
 *
 *  @ingroup Simulator/Edges/NG911
 *
 *  @brief Specialization of the AllEdges class for the NG911 network
 */

#include "All911Edges.h"
#include "Book.h"

__global__ void advance911EdgesDevice();

///  Allocate GPU memories to store all edge states,
///  and copy them from host to GPU memory.
///
///  @param  allEdgesDevice  GPU address of the All911EdgeDeviceProperties struct
///                             on device memory.
void All911Edges::allocEdgeDeviceStruct(void **allEdgesDevice)
{
   allocEdgeDeviceStruct(allEdgesDevice, Simulator::getInstance().getTotalVertices(), Simulator::getInstance().getMaxEdgesPerVertex());
}

///  Allocate GPU memories to store all edge states,
///  and copy them from host to GPU memory.
///
///  @param  allEdgesDevice     GPU address of the All911EdgeDeviceProperties struct
///                                on device memory.
///  @param  numVertices            Number of vertices.
///  @param  maxEdgesPerVertex  Maximum number of edges per vertex.
void All911Edges::allocEdgeDeviceStruct(void **allEdgesDevice, int numVertices, int maxEdgesPerVertex)
{
   All911EdgeDeviceProperties allEdges;
   allocDeviceStruct(allEdges, numVertices, maxEdgesPerVertex);
   HANDLE_ERROR(cudaMalloc(allEdgesDevice, sizeof(All911EdgeDeviceProperties)));
   HANDLE_ERROR(cudaMemcpy(*allEdgesDevice, &allEdges, sizeof(All911EdgeDeviceProperties), cudaMemcpyHostToDevice));
}

///  Allocate GPU memories to store all synapses' states,
///  and copy them from host to GPU memory.
///  (Helper function of allocEdgeDeviceStruct)
///
///  @param  allEdgesDevice     GPU address of the All911EdgeDeviceProperties struct
///                                on device memory.
///  @param  numVertices            Number of vertices.
///  @param  maxEdgesPerVertex  Maximum number of edges per vertex.
void All911Edges::allocDeviceStruct(All911EdgeDeviceProperties &allEdgesDevice, int numVertices, int maxEdgesPerVertex)
{
   BGSIZE maxTotalEdges = maxEdgesPerVertex * numVertices;
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.sourceVertexIndex_, maxTotalEdges * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.destVertexIndex_, maxTotalEdges * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.W_, maxTotalEdges * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.type_, maxTotalEdges * sizeof(edgeType)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.inUse_, maxTotalEdges * sizeof(unsigned char)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.edgeCounts_, numVertices * sizeof(BGSIZE)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.isAvailable_, maxTotalEdges * sizeof(unsigned char)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.isRedial_, maxTotalEdges * sizeof(unsigned char)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.vertexId_, maxTotalEdges * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.time_, maxTotalEdges * sizeof(uint64_t)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.duration_, maxTotalEdges * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.x_, maxTotalEdges * sizeof(double)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.y_, maxTotalEdges * sizeof(double)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.patience_, maxTotalEdges * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.onSiteTime_, maxTotalEdges * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.responderType_, maxTotalEdges * sizeof(ResponderTypes)));
}

///  Delete GPU memories.
///
///  @param  allEdgesDevice  GPU address of the All911EdgeDeviceProperties struct
///                             on device memory.
void All911Edges::deleteEdgeDeviceStruct(void *allEdgesDevice)
{
   All911EdgeDeviceProperties allEdgesDeviceProps;
   HANDLE_ERROR(cudaMemcpy(&allEdgesDeviceProps, allEdgesDevice, sizeof(All911EdgeDeviceProperties), cudaMemcpyDeviceToHost));
   deleteDeviceStruct(allEdgesDeviceProps);
   HANDLE_ERROR(cudaFree(allEdgesDevice));
}

///  Delete GPU memories.
///  (Helper function of deleteEdgeDeviceStruct)
///
///  @param  allEdgesDeviceProps  GPU address of the All911EdgeDeviceProperties struct
///                             on device memory.
void All911Edges::deleteDeviceStruct(All911EdgeDeviceProperties &allEdgesDeviceProps)
{
   HANDLE_ERROR(cudaFree(allEdgesDeviceProps.sourceVertexIndex_));
   HANDLE_ERROR(cudaFree(allEdgesDeviceProps.destVertexIndex_));
   HANDLE_ERROR(cudaFree(allEdgesDeviceProps.W_));
   HANDLE_ERROR(cudaFree(allEdgesDeviceProps.type_));
   HANDLE_ERROR(cudaFree(allEdgesDeviceProps.inUse_));
   HANDLE_ERROR(cudaFree(allEdgesDeviceProps.edgeCounts_));
   HANDLE_ERROR(cudaFree(AllEdgesDeviceProps.isAvailable_));
   HANDLE_ERROR(cudaFree(AllEdgesDeviceProps.isRedial_));
   HANDLE_ERROR(cudaFree(AllEdgesDeviceProps.vertexId_));
   HANDLE_ERROR(cudaFree(AllEdgesDeviceProps.time_));
   HANDLE_ERROR(cudaFree(AllEdgesDeviceProps.duration_));
   HANDLE_ERROR(cudaFree(AllEdgesDeviceProps.x_));
   HANDLE_ERROR(cudaFree(AllEdgesDeviceProps.y_));
   HANDLE_ERROR(cudaFree(AllEdgesDeviceProps.patience_));
   HANDLE_ERROR(cudaFree(AllEdgesDeviceProps.onSiteTime_));
   HANDLE_ERROR(cudaFree(AllEdgesDeviceProps.responderType_));
}

///  Copy all edge data from host to device.
///
///  @param  allEdgesDevice  GPU address of the All911EdgeDeviceProperties struct
///                             on device memory.
void All911Edges::copyEdgeHostToDevice(void *allEdgesDevice)
{
   copyEdgeHostToDevice(allEdgesDevice, Simulator::getInstance().getTotalVertices(), Simulator::getInstance().getMaxEdgesPerVertex());
}

///  Copy all edge data from host to device.
///
///  @param  allEdgesDevice     GPU address of the All911EdgeDeviceProperties struct
///                                on device memory.
///  @param  numVertices            Number of vertices.
///  @param  maxEdgesPerVertex  Maximum number of edges per vertex.
void All911Edges::copyEdgeHostToDevice(void *allEdgesDevice, int numVertices, int maxEdgesPerVertex)
{   // copy everything necessary
   All911EdgeDeviceProperties allEdgesDeviceProps;
   HANDLE_ERROR(cudaMemcpy(&allEdgesDeviceProps, allEdgesDevice, sizeof(All911EdgeDeviceProperties), cudaMemcpyDeviceToHost));
   copyHostToDevice(allEdgesDevice, allEdgesDeviceProps, numVertices, maxEdgesPerVertex);
}

///  Copy all synapses' data from host to device.
///  (Helper function of copyEdgeHostToDevice)
///
///  @param  allEdgesDevice           GPU address of the All911EdgeDeviceProperties struct on device memory.
///  @param  allEdgesDeviceProps      CPU address of the All911EdgeDeviceProperties struct on host memory.
///  @param  numVertices              Number of vertices.
///  @param  maxEdgesPerVertex        Maximum number of edges per vertex.
void All911Edges::copyHostToDevice(void *allEdgesDevice, All911EdgeDeviceProperties &allEdgesDeviceProps, int numVertices, int maxEdgesPerVertex)
{
   BGSIZE maxTotalEdges = maxEdgesPerVertex * numVertices;
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.sourceVertexIndex_, sourceVertexIndex_.data(), maxTotalEdges * sizeof(int), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.destVertexIndex_, destVertexIndex_.data(), maxTotalEdges * sizeof(int), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.W_, W_.data(), maxTotalEdges * sizeof(BGFLOAT), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.type_, type_.data(), maxTotalEdges * sizeof(edgeType), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.inUse_, inUse_.data(), maxTotalEdges * sizeof(unsigned char), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.edgeCounts_, edgeCounts_.data(), numVertices * sizeof(BGSIZE), cudaMemcpyHostToDevice));
   allEdgesDeviceProps.totalEdgeCount_ = totalEdgeCount_;
   allEdgesDeviceProps.maxEdgesPerVertex_ = maxEdgesPerVertex_;
   allEdgesDeviceProps.countVertices_ = countVertices_;
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.isAvailable_, isAvailable_.data(), maxTotalEdges * sizeof(unsigned char), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.isRedial_, isRedial_.data(), maxTotalEdges * sizeof(unsigned char), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.vertexId_, vertexId_.data(), maxTotalEdges * sizeof(int), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.time_, time_.data(), maxTotalEdges * sizeof(uint64_t), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.duration_, duration_.data(), maxTotalEdges * sizeof(int), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.x_, x_.data(), maxTotalEdges * sizeof(double), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.y_, y_.data(), maxTotalEdges * sizeof(double), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.patience_, patience_.data(), maxTotalEdges * sizeof(int), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.onSiteTime_, onSiteTime_.data(), maxTotalEdges * sizeof(int), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.responderType_, responderType_.data(), maxTotalEdges * sizeof(ResponderTypes), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDevice, &allEdgesDeviceProps, sizeof(All911EdgeDeviceProperties), cudaMemcpyHostToDevice));
   // Set countVertices_ to 0 to avoid illegal memory deallocation
   // at All911Edges deconstructor.
   allEdgesDeviceProps.countVertices_ = 0;
}

///  Copy all edge data from device to host.
///
///  @param  allEdgesDevice  GPU address of the All911EdgeDeviceProperties struct
///                             on device memory.
void All911Edges::copyEdgeDeviceToHost(void *allEdgesDevice)
{
   // copy everything necessary
   All911EdgeDeviceProperties allEdgesDeviceProps;
   HANDLE_ERROR(cudaMemcpy(&allEdgesDeviceProps, allEdgesDevice, sizeof(All911EdgeDeviceProperties), cudaMemcpyDeviceToHost));
   copyDeviceToHost(allEdgesDeviceProps);
}

///  Copy all edge data from device to host.
///  (Helper function of copyEdgeDeviceToHost)
///
///  @param  allEdgesProperties     GPU address of the All911EdgeDeviceProperties struct
///                                on device memory.
void All911Edges::copyDeviceToHost(All911EdgeDeviceProperties &allEdgesDeviceProps)
{
   int numVertices = Simulator::getInstance().getTotalVertices();
   BGSIZE maxTotalEdges = Simulator::getInstance().getMaxEdgesPerVertex() * numVertices;
   HANDLE_ERROR(cudaMemcpy(sourceVertexIndex_.data(), allEdgesDeviceProps.sourceVertexIndex_, maxTotalEdges * sizeof(int), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(destVertexIndex_.data(), allEdgesDeviceProps.destVertexIndex_, maxTotalEdges * sizeof(int), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(W_.data(), allEdgesDeviceProps.W_, maxTotalEdges * sizeof(BGFLOAT), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(type_.data(), allEdgesDeviceProps.type_, maxTotalEdges * sizeof(edgeType), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(inUse_.data(), allEdgesDeviceProps.inUse_, maxTotalEdges * sizeof(unsigned char), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(edgeCounts_.data(), allEdgesDeviceProps.edgeCounts_, numVertices * sizeof(BGSIZE), cudaMemcpyDeviceToHost));
   totalEdgeCount_ = allEdgesDeviceProps.totalEdgeCount_;
   maxEdgesPerVertex_ = allEdgesDeviceProps.maxEdgesPerVertex_;
   countVertices_ = allEdgesDeviceProps.countVertices_;
   // Set countVertices_ to 0 to avoid illegal memory deallocation
   // at All911Edges deconstructor.
   allEdgesDeviceProps.countVertices_ = 0;
   HANDLE_ERROR(cudaMemcpy(isAvailable_.data(), allEdgesDeviceProps.isAvailable_, maxTotalEdges * sizeof(unsigned char), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(isRedial_.data(), allEdgesDeviceProps.isRedial_, maxTotalEdges * sizeof(unsigned char), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(vertexId_.data(), allEdgesDeviceProps.vertexId_, maxTotalEdges * sizeof(int), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(time_.data(), allEdgesDeviceProps.time_, maxTotalEdges * sizeof(uint64_t), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(duration_.data(), allEdgesDeviceProps.duration_, maxTotalEdges * sizeof(int), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(x_.data(), allEdgesDeviceProps.x_, maxTotalEdges * sizeof(double), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(y_.data(), allEdgesDeviceProps.y_, maxTotalEdges * sizeof(double), cudamemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(patience_.data(), allEdgesDeviceProps.patience_, maxTotalEdges * sizeof(int), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(onSiteTime_.data(), allEdgesDeviceProps.onSiteTime_, maxTotalEdges * sizeof(int), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(responderType_.data(), allEdgesDeviceProps.responderType_, maxTotalEdges * sizeof(ResponderTypes), cudaMemcpyDeviceToHost));
}

///  Get edge_counts in AllEdges struct on device memory.
///
///  @param  allEdgesDevice  GPU address of the All911EdgeDeviceProperties struct
///                             on device memory.
void All911Edges::copyDeviceEdgeCountsToHost(void *allEdgesDevice)
{
   All911EdgeDeviceProperties allEdgesDeviceProps;
   int vertexCount = Simulator::getInstance().getTotalVertices();
   HANDLE_ERROR(cudaMemcpy(&allEdgesDeviceProps, allEdgesDevice, sizeof(All911EdgeDeviceProperties), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(edgeCounts_.data(), allEdgesDeviceProps.edgeCounts_, vertexCount * sizeof(BGSIZE), cudaMemcpyDeviceToHost));
}

//*********Question: is this a neuro implementation? Not sure if edge summation is used for 911 implementation
//*****************  but should consider refactoring it out of AllEdges.h and into AllNeuroEdges.h 
///  Get summationCoord and in_use in AllEdges struct on device memory.
///
///  @param  allEdgesDevice  GPU address of the All911EdgeDeviceProperties struct
///                             on device memory.
void All911Edges::copyDeviceEdgeSumIdxToHost(void *allEdgesDevice)
{
   All911EdgeDeviceProperties allEdgesDeviceProps;
   BGSIZE maxTotalEdges = Simulator::getInstance().getMaxEdgesPerVertex() * Simulator::getInstance().getTotalVertices();
   HANDLE_ERROR(cudaMemcpy(&allEdgesDeviceProps, allEdgesDevice, sizeof(All911EdgeDeviceProperties), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(sourceVertexIndex_.data(), allEdgesDeviceProps.sourceVertexIndex_, maxTotalEdges * sizeof(int), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(inUse_.data(), allEdgesDeviceProps.inUse_, maxTotalEdges * sizeof(unsigned char), cudaMemcpyDeviceToHost));
}

// ///  Advance all the edges in the simulation.
// ///  Update the state of all edges for a time step.
// ///
// ///  @param  allEdgesDevice      GPU address of the AllEdgesDeviceProperties struct
// ///                                 on device memory.
// ///  @param  allVerticesDevice       GPU address of the AllVerticesDeviceProperties struct on device memory.
// ///  @param  edgeIndexMapDevice  GPU address of the EdgeIndexMap on device memory.
// void All911Edges::advanceEdges(void *allEdgesDevice, void *allVerticesDevice, void *edgeIndexMapDevice)
// {
//    if (totalEdgeCount_ == 0)
//       return;
//    // CUDA parameters
//    const int threadsPerBlock = 256;
//    int blocksPerGrid = (totalEdgeCount_ + threadsPerBlock - 1) / threadsPerBlock;
//    // Advance synapses ------------->
//    advance911EdgesDevice<<<blocksPerGrid, threadsPerBlock>>>(
//       totalEdgeCount_, (EdgeIndexMapDevice *)edgeIndexMapDevice, g_simulationStep,
//       Simulator::getInstance().getDeltaT(), (All911EdgeDeviceProperties *)allEdgesDevice);
// }

///  Advance all the edges in the simulation.
void All911Edges::advanceEdges(void *allEdgesDevice, void *allVerticesDevice, void *edgeIndexMapDevice)
{
   if (totalEdgeCount_ == 0)
      return;
   // CUDA parameters
   const int threadsPerBlock = 256;
   int blocksPerGrid = (totalEdgeCount_ + threadsPerBlock - 1) / threadsPerBlock;

   //Advancement logic
   Simulator &simulator = Simulator::getInstance();
   //What is this if we haven't implemented a GPU version of vertices??
   AllVertices *vertices = (AllVertices *)allVerticesDevice;
   All911Vertices &all911Vertices = dynamic_cast<All911Vertices &>(*vertices);

   for (int vertex = 0; vertex < simulator.getTotalVertices(); ++vertex) {
      if (simulator.getModel().getLayout().vertexTypeMap_[vertex] == CALR) {
         continue;   // TODO911: Caller Regions will have different behaviour
      }
      advanceSingleEdge(vertex);
   }
}

void All911Edges::advanceSingleEdge(int vertex) {
   int start = edgeIndexMap.incomingEdgeBegin_[vertex];
   int count = edgeIndexMap.incomingEdgeCount_[vertex];

   // Loop over all the edges and pull the data in
   for (int eIdxMap = start; eIdxMap < start + count; ++eIdxMap) {
      int edgeIdx = edgeIndexMap.incomingEdgeIndexMap_[eIdxMap];

      if (!inUse_[edgeIdx]) {
         continue;
      }   // Edge isn't in use
      if (isAvailable_[edgeIdx]) {
         continue;
      }   // Edge doesn't have a call

      int dst = destVertexIndex_[edgeIdx];
      // The destination vertex should be the one pulling the information
      assert(dst == vertex);

      CircularBuffer<Call> &dstQueue = all911Vertices.getQueue(dst);
      if (dstQueue.size() >= (dstQueue.capacity() - all911Vertices.busyServers(dst))) {
         // Call is dropped because there is no space in the waiting queue
         if (!isRedial_[edgeIdx]) {
            // Only count the dropped call if it's not a redial
            all911Vertices.droppedCalls(dst)++;
            // Record that we received a call
            all911Vertices.receivedCalls(dst)++;
            LOG4CPLUS_DEBUG(edgeLogger_,
                           "Call dropped: " << all911Vertices.droppedCalls(dst) << ", time: "
                                             << time_[edgeIdx] << ", vertex: " << dst
                                             << ", queue size: " << dstQueue.size());
         }
      } else {
         // Transfer call to destination
         dstQueue.put(call_[edgeIdx]);
         // Record that we received a call
         all911Vertices.receivedCalls(dst)++;
         isAvailable_[edgeIdx] = true;
         isRedial_[edgeIdx] = false;
      }
   }
}

///  Set some parameters used for advanceEdgesDevice.
void All911Edges::setAdvanceEdgesDeviceParams()
{
   setEdgeClassID();
}

///  Set ESCS class ID defined by ESCSEdges for the caller's edge class.
///  The class ID will be set to escsClass_d in device memory,
///  and the escsClass_d will be referred to call a device function for the
///  particular edge class.
///  Because we cannot use virtual function (Polymorphism) in device functions,
///  we use this scheme.
///  Note: we used to use a function pointer; however, it caused the growth_cuda crash
///  (see issue#137).
void All911Edges::setEdgeClassID()
{
   ESCSEdges escsClass_h = NineOneOneEdges;
   HANDLE_ERROR(cudaMemcpyToSymbol(escsClass_d, &escsClass_h, sizeof(ESCSEdges)));
}

///  Prints GPU 911EdgesProps data.
///
///  @param  allEdgesDeviceProps   GPU address of the corresponding All911EdgeDeviceProperties struct on device memory.
void All911Edges::printGPUEdgesProps(void *allEdgesDeviceProps) const
{
   All911EdgeDeviceProperties allEdgesProps;
   BGSIZE size = maxEdgesPerVertex_ * countVertices_;
   if (size != 0) {
      //allocate print out data members
      int *sourceVertexIndexPrint = new int[size];
      int *destVertexIndexPrint = new int[size];
      BGFLOAT *WPrint = new BGFLOAT[size];
      edgeType *typePrint = new edgeType[size];
      // The representation of inUsePrint has been updated from bool to unsigned char
      // to store 1 (true) or 0 (false) for the support of serialization operations. See ISSUE-459
      unsigned char *inUsePrint = new unsigned char[size];
      BGSIZE *edgeCountsPrint = new BGSIZE[countVertices_];
      BGSIZE totalEdgeCountPrint;
      BGSIZE maxEdgesPerVertexPrint;
      int countVerticesPrint;
      unsigned char *isAvailablePrint = new unsigned char[size];
      unsigned char *isRedialPrint = new unsigned char[size];
      int *vertexIdPrint = new int[size];
      uint64_t *timePrint = new uint64_t[size];
      int *durationPrint = new int[size];
      double *xPrint = new double[size];
      double *yPrint = new double[size];
      int *patiencePrint = new int[size];
      int *onSiteTimePrint = new int[size];
      ResponderTypes *responderTypePrint = new ResponderTypes[size];

      //set some array to default values
      //TODO: should look into why this is necessary
      for (BGSIZE i = 0; i < size; i++) {
         inUsePrint[i] = false;
      }

      for (int i = 0; i < countVertices_; i++) {
         edgeCountsPrint[i] = 0;
      }

      // copy everything
      HANDLE_ERROR(cudaMemcpy(&allEdgesProps, allEdgesDeviceProps, sizeof(All911EdgeDeviceProperties), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(sourceVertexIndexPrint, allEdgesProps.sourceVertexIndex_, size * sizeof(int), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(destVertexIndexPrint, allEdgesProps.destVertexIndex_, size * sizeof(int), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(WPrint, allEdgesProps.W_, size * sizeof(BGFLOAT), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(typePrint, allEdgesProps.type_, size * sizeof(edgeType), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(inUsePrint, allEdgesProps.inUse_, size * sizeof(unsigned char), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(edgeCountsPrint, allEdgesProps.edgeCounts_, countVertices_ * sizeof(BGSIZE), cudaMemcpyDeviceToHost));
      totalEdgeCountPrint = allEdgesProps.totalEdgeCount_;
      maxEdgesPerVertexPrint = allEdgesProps.maxEdgesPerVertex_;
      countVerticesPrint = allEdgesProps.countVertices_;
      // Set countVertices_ to 0 to avoid illegal memory deallocation
      // at AllSynapsesProps deconstructor.
      allSynapsesProps.countVertices_ = 0;
      HANDLE_ERROR(cudaMemcpy(isAvailablePrint, allEdgesProps.isAvailable_, size * sizeof(unsigned char), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(isRedialPrint, allEdgesProps.isRedial_, size * sizeof(unsigned char), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(vertexIdPrint, allEdgesProps.vertexId_, size * sizeof(int), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(timePrint, allEdgesProps.time_, size * sizeof(uint64_t), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(durationPrint, allEdgesProps.duration_, size * sizeof(int), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(xPrint, allEdgesProps.x_, size * sizeof(double), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(yPrint, allEdgesProps.y_, size * sizeof(double), cudamemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(patiencePrint, allEdgesProps.patience_, size * sizeof(int), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(onSiteTimePrint, allEdgesProps.onSiteTime_, size * sizeof(int), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(responderTypePrint, allEdgesProps.responderType_, size * sizeof(ResponderTypes), cudaMemcpyDeviceToHost));

      //Print everything
      for (BGSIZE i = 0; i < size; i++) {
         if (WPrint[i] ! = 0.0) {
            cout << "GPU W[" << i << "] = " << WPrint[i];
            cout << " GPU sourceVertexIndex: " << sourceVertexIndexPrint[i];
            cout << " GPU destVertexIndex: " << destVertexIndexPrint[i];
            cout << " GPU type: " << typePrint[i];
            cout << " GPU inUse: " << (inUsePrint[i] == 1 ? "true" : "false");
            cout << " GPU isAvailable: " << (isAvailablePrint[i] == 1 ? "true" : "false");
            cout << " GPU isRedial: " << (isRedialPrint[i] == 1 ? "true" : "false");
            cout << " GPU eventVertexIndex: " << vertexIdPrint[i];
            cout << " GPU eventTime: " << timePrint[i];
            cout << " GPU eventDuration: " << durationPrint[i];
            cout << " GPU eventLocationX: " << xPrint[i];
            cout << " GPU eventLocationY: " << yPrint[i];
            cout << " GPU customerPatience: " << patiencePrint[i];
            cout << " GPU responderOnSiteTime: " << onSiteTimePrint[i];
            cout << " GPU responderType: " << responderTypePrint[i];
         }
      }
      for (int i = 0; i < countVertices_; i++) {
         cout << "GPU edgeCounts: " << "vertex[" << i << "]" << edgeCountsPrint[i] << endl;
      }
      cout << "GPU totalEdgeCount: " << totalEdgeCountPrint << endl;
      cout << "GPU maxEdgesPerVertex: " << maxEdgesPerVertexPrint << endl;
      cout << "GPU countVertices: " << countVerticesPrint << endl;

      //Clean up everything
      delete[] sourceVertexIndexPrint;
      sourceVertexIndexPrint = nullptr;

      delete[] destVertexIndexPrint;
      destVertexIndexPrint = nullptr;

      delete[] WPrint;
      WPrint = nullptr;

      delete[] typePrint;
      typePrint = nullptr;

      delete[] inUsePrint;
      inUsePrint = nullptr;

      delete[] edgeCountsPrint;
      edgeCountsPrint = nullptr;

      delete[] isAvailablePrint;
      isAvailablePrint = nullptr;

      delete[] isRedialPrint;
      isRedialPrint = nullptr;

      delete[] vertexIdPrint;
      vertexIdPrint = nullptr;

      delete[] timePrint;
      timePrint = nullptr;

      delete[] durationPrint;
      durationPrint = nullptr;

      delete[] xPrint;
      xPrint = nullptr;

      delete[] yPrint;
      yPrint = nullptr;

      delete[] patiencePrint;
      patiencePrint = nullptr;

      delete[] onSiteTimePrint;
      onSiteTimePrint = nullptr;

      delete[] responderTypePrint;
      responderTypePrint = nullptr;
   }
}

__global__ void advance911EdgesDevice(int totalEdgeCount, EdgeIndexMapDevice *edgeIndexMapDevice, All911EdgeDeviceProperties *allEdgesDevice)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx >= totalEdgeCount) {
      return;
   }
   BGSIZE iEdg = edgeIndexMapDevice->incomingEdgeIndexMap_[idx];
   unsigned char inUse = allEdgesDevice->inUse_[iEdg];
   unsigned char isAvailable = allEdgesDevice->isAvailable_[iEdg];
   unsigned char isRedial = allEdgesDevice->isRedial_[iEdg];

   if (inUse == 0) {
      return;
   }   // Edge isn't in use
   if (isAvailable == 1) {
      return;
   }   // Edge doesn't have a call
   
   int dst = allEdgesDevice->destVertexIndex_[iEdg];
   // The destination vertex should be the one pulling the information
   //Not sure if assert can be called from a kernel
   //assert(dst == vertex);
}