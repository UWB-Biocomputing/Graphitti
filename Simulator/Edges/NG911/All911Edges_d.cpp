/**
 *  @file All911Edges_d.cpp
 *
 *  @ingroup Simulator/Edges/NG911
 *
 *  @brief Specialization of the AllEdges class for the NG911 network
 */

#include "All911Edges.h"
#include "Book.h"
#include "GPUModel.h"

///  Allocate GPU memories to store all edge states,
///  and copy them from host to GPU memory.
///
///  @param  allEdgesDevice  GPU address of the All911EdgesDeviceProperties struct
///                             on device memory.
void All911Edges::allocEdgeDeviceStruct()
{
   GPUModel *gpuModel = static_cast<GPUModel *>(&Simulator::getInstance().getModel());
   void **allEdgesDevice = reinterpret_cast<void **>(&(gpuModel->getAllEdgesDevice()));
   allocEdgeDeviceStruct(allEdgesDevice, Simulator::getInstance().getTotalVertices(),
                         Simulator::getInstance().getMaxEdgesPerVertex());
}

///  Allocate GPU memories to store all edge states,
///  and copy them from host to GPU memory.
///
///  @param  allEdgesDevice     GPU address of the All911EdgesDeviceProperties struct
///                                on device memory.
///  @param  numVertices            Number of vertices.
///  @param  maxEdgesPerVertex  Maximum number of edges per vertex.
void All911Edges::allocEdgeDeviceStruct(void **allEdgesDevice, int numVertices,
                                        int maxEdgesPerVertex)
{
   All911EdgesDeviceProperties allEdges;
   allocDeviceStruct(allEdges, numVertices, maxEdgesPerVertex);
   HANDLE_ERROR(cudaMalloc(allEdgesDevice, sizeof(All911EdgesDeviceProperties)));
   HANDLE_ERROR(cudaMemcpy(*allEdgesDevice, &allEdges, sizeof(All911EdgesDeviceProperties),
                           cudaMemcpyHostToDevice));
}

///  Allocate GPU memories to store all edges' states,
///  and copy them from host to GPU memory.
///  (Helper function of allocEdgeDeviceStruct)
///
///  @param  allEdgesDevice     GPU address of the All911EdgesDeviceProperties struct
///                                on device memory.
///  @param  numVertices            Number of vertices.
///  @param  maxEdgesPerVertex  Maximum number of edges per vertex.
void All911Edges::allocDeviceStruct(All911EdgesDeviceProperties &allEdgesDevice, int numVertices,
                                    int maxEdgesPerVertex)
{
   BGSIZE maxTotalEdges = maxEdgesPerVertex * numVertices;
   HANDLE_ERROR(
      cudaMalloc((void **)&allEdgesDevice.sourceVertexIndex_, maxTotalEdges * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.destVertexIndex_, maxTotalEdges * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.W_, maxTotalEdges * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.type_, maxTotalEdges * sizeof(edgeType)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.inUse_, maxTotalEdges * sizeof(unsigned char)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.edgeCounts_, numVertices * sizeof(BGSIZE)));
   HANDLE_ERROR(
      cudaMalloc((void **)&allEdgesDevice.isAvailable_, maxTotalEdges * sizeof(unsigned char)));
   HANDLE_ERROR(
      cudaMalloc((void **)&allEdgesDevice.isRedial_, maxTotalEdges * sizeof(unsigned char)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.vertexId_, maxTotalEdges * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.time_, maxTotalEdges * sizeof(uint64_t)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.duration_, maxTotalEdges * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.x_, maxTotalEdges * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.y_, maxTotalEdges * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.patience_, maxTotalEdges * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.onSiteTime_, maxTotalEdges * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.responderType_, maxTotalEdges * sizeof(int)));
}

///  Delete GPU memories.
///
///  @param  allEdgesDevice  GPU address of the All911EdgesDeviceProperties struct
///                             on device memory.
void All911Edges::deleteEdgeDeviceStruct()
{
   All911EdgesDeviceProperties allEdgesDeviceProps;
   GPUModel *gpuModel = static_cast<GPUModel *>(&Simulator::getInstance().getModel());
   void *allEdgesDevice = static_cast<void *>(gpuModel->getAllEdgesDevice());
   HANDLE_ERROR(cudaMemcpy(&allEdgesDeviceProps, allEdgesDevice,
                           sizeof(All911EdgesDeviceProperties), cudaMemcpyDeviceToHost));
   deleteDeviceStruct(allEdgesDeviceProps);
   HANDLE_ERROR(cudaFree(allEdgesDevice));
}

///  Delete GPU memories.
///  (Helper function of deleteEdgeDeviceStruct)
///
///  @param  allEdgesDeviceProps  GPU address of the All911EdgesDeviceProperties struct
///                             on device memory.
void All911Edges::deleteDeviceStruct(All911EdgesDeviceProperties &allEdgesDeviceProps)
{
   HANDLE_ERROR(cudaFree(allEdgesDeviceProps.sourceVertexIndex_));
   HANDLE_ERROR(cudaFree(allEdgesDeviceProps.destVertexIndex_));
   HANDLE_ERROR(cudaFree(allEdgesDeviceProps.W_));
   HANDLE_ERROR(cudaFree(allEdgesDeviceProps.type_));
   HANDLE_ERROR(cudaFree(allEdgesDeviceProps.inUse_));
   HANDLE_ERROR(cudaFree(allEdgesDeviceProps.edgeCounts_));
   HANDLE_ERROR(cudaFree(allEdgesDeviceProps.isAvailable_));
   HANDLE_ERROR(cudaFree(allEdgesDeviceProps.isRedial_));
   HANDLE_ERROR(cudaFree(allEdgesDeviceProps.vertexId_));
   HANDLE_ERROR(cudaFree(allEdgesDeviceProps.time_));
   HANDLE_ERROR(cudaFree(allEdgesDeviceProps.duration_));
   HANDLE_ERROR(cudaFree(allEdgesDeviceProps.x_));
   HANDLE_ERROR(cudaFree(allEdgesDeviceProps.y_));
   HANDLE_ERROR(cudaFree(allEdgesDeviceProps.patience_));
   HANDLE_ERROR(cudaFree(allEdgesDeviceProps.onSiteTime_));
   HANDLE_ERROR(cudaFree(allEdgesDeviceProps.responderType_));
}

///  Copy all edge data from host to device.
///
///  @param  allEdgesDevice  GPU address of the All911EdgesDeviceProperties struct
///                             on device memory.
void All911Edges::copyEdgeHostToDevice()
{
   GPUModel *gpuModel = static_cast<GPUModel *>(&Simulator::getInstance().getModel());
   void *allEdgesDevice = static_cast<void *>(gpuModel->getAllEdgesDevice());
   copyEdgeHostToDevice(allEdgesDevice, Simulator::getInstance().getTotalVertices(),
                        Simulator::getInstance().getMaxEdgesPerVertex());
}

///  Copy all edge data from host to device.
///
///  @param  allEdgesDevice     GPU address of the All911EdgesDeviceProperties struct
///                                on device memory.
///  @param  numVertices            Number of vertices.
///  @param  maxEdgesPerVertex  Maximum number of edges per vertex.
void All911Edges::copyEdgeHostToDevice(void *allEdgesDevice, int numVertices, int maxEdgesPerVertex)
{   // copy everything necessary
   All911EdgesDeviceProperties allEdgesDeviceProps;
   HANDLE_ERROR(cudaMemcpy(&allEdgesDeviceProps, allEdgesDevice,
                           sizeof(All911EdgesDeviceProperties), cudaMemcpyDeviceToHost));
   copyHostToDevice(allEdgesDevice, allEdgesDeviceProps, numVertices, maxEdgesPerVertex);
}

///  Copy all edges' data from host to device.
///  (Helper function of copyEdgeHostToDevice)
///
///  @param  allEdgesDevice           GPU address of the All911EdgesDeviceProperties struct on device memory.
///  @param  allEdgesDeviceProps      CPU address of the All911EdgesDeviceProperties struct on host memory.
///  @param  numVertices              Number of vertices.
///  @param  maxEdgesPerVertex        Maximum number of edges per vertex.
void All911Edges::copyHostToDevice(void *allEdgesDevice,
                                   All911EdgesDeviceProperties &allEdgesDeviceProps,
                                   int numVertices, int maxEdgesPerVertex)
{
   BGSIZE maxTotalEdges = maxEdgesPerVertex * numVertices;
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.sourceVertexIndex_, sourceVertexIndex_.data(),
                           maxTotalEdges * sizeof(int), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.destVertexIndex_, destVertexIndex_.data(),
                           maxTotalEdges * sizeof(int), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.W_, W_.data(), maxTotalEdges * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.type_, type_.data(),
                           maxTotalEdges * sizeof(edgeType), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.inUse_, inUse_.data(),
                           maxTotalEdges * sizeof(unsigned char), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.edgeCounts_, edgeCounts_.data(),
                           numVertices * sizeof(BGSIZE), cudaMemcpyHostToDevice));
   allEdgesDeviceProps.totalEdgeCount_ = totalEdgeCount_;
   allEdgesDeviceProps.maxEdgesPerVertex_ = maxEdgesPerVertex_;
   allEdgesDeviceProps.countVertices_ = countVertices_;
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.isAvailable_, isAvailable_.data(),
                           maxTotalEdges * sizeof(unsigned char), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.isRedial_, isRedial_.data(),
                           maxTotalEdges * sizeof(unsigned char), cudaMemcpyHostToDevice));

   // Bracket array declaration to memcpy to manually release array from stack
   // This is necessary to prevent segmentation faults when running large graphs
   {
      int cpuVertexId[maxTotalEdges];
      for (int i = 0; i < maxTotalEdges; i++) {
         cpuVertexId[i] = call_[i].vertexId;
      }
      HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.vertexId_, cpuVertexId,
                              maxTotalEdges * sizeof(int), cudaMemcpyHostToDevice));
   }

   // Bracket array declaration to memcpy to manually release array from stack
   // This is necessary to prevent segmentation faults when running large graphs
   {
      uint64_t cpuTime[maxTotalEdges];
      for (int i = 0; i < maxTotalEdges; i++) {
         cpuTime[i] = call_[i].time;
      }
      HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.time_, cpuTime, maxTotalEdges * sizeof(uint64_t),
                              cudaMemcpyHostToDevice));
   }

   // Bracket array declaration to memcpy to manually release array from stack
   // This is necessary to prevent segmentation faults when running large graphs
   {
      int cpuDuration[maxTotalEdges];
      for (int i = 0; i < maxTotalEdges; i++) {
         cpuDuration[i] = call_[i].duration;
      }
      HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.duration_, cpuDuration,
                              maxTotalEdges * sizeof(int), cudaMemcpyHostToDevice));
   }

   // Bracket array declaration to memcpy to manually release array from stack
   // This is necessary to prevent segmentation faults when running large graphs
   {
      BGFLOAT cpuX[maxTotalEdges];
      for (int i = 0; i < maxTotalEdges; i++) {
         cpuX[i] = call_[i].x;
      }
      HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.x_, cpuX, maxTotalEdges * sizeof(BGFLOAT),
                              cudaMemcpyHostToDevice));
   }

   // Bracket array declaration to memcpy to manually release array from stack
   // This is necessary to prevent segmentation faults when running large graphs
   {
      BGFLOAT cpuY[maxTotalEdges];
      for (int i = 0; i < maxTotalEdges; i++) {
         cpuY[i] = call_[i].y;
      }
      HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.y_, cpuY, maxTotalEdges * sizeof(BGFLOAT),
                              cudaMemcpyHostToDevice));
   }

   // Bracket array declaration to memcpy to manually release array from stack
   // This is necessary to prevent segmentation faults when running large graphs
   {
      int cpuPatience[maxTotalEdges];
      for (int i = 0; i < maxTotalEdges; i++) {
         cpuPatience[i] = call_[i].patience;
      }
      HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.patience_, cpuPatience,
                              maxTotalEdges * sizeof(int), cudaMemcpyHostToDevice));
   }

   // Bracket array declaration to memcpy to manually release array from stack
   // This is necessary to prevent segmentation faults when running large graphs
   {
      int cpuOnSiteTime[maxTotalEdges];
      for (int i = 0; i < maxTotalEdges; i++) {
         cpuOnSiteTime[i] = call_[i].onSiteTime;
      }
      HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.onSiteTime_, cpuOnSiteTime,
                              maxTotalEdges * sizeof(int), cudaMemcpyHostToDevice));
   }

   // Bracket array declaration to memcpy to manually release array from stack
   // This is necessary to prevent segmentation faults when running large graphs
   {
      int cpuResponderType[maxTotalEdges];
      for (int i = 0; i < maxTotalEdges; i++) {
         if (call_[i].type == "Law")
            cpuResponderType[i] = 7;
         else if (call_[i].type == "EMS")
            cpuResponderType[i] = 5;
         else if (call_[i].type == "Fire")
            cpuResponderType[i] = 6;
      }
      HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.responderType_, cpuResponderType,
                              maxTotalEdges * sizeof(int), cudaMemcpyHostToDevice));
   }

   HANDLE_ERROR(cudaMemcpy(allEdgesDevice, &allEdgesDeviceProps,
                           sizeof(All911EdgesDeviceProperties), cudaMemcpyHostToDevice));
   // Set countVertices_ to 0 to avoid illegal memory deallocation
   // at All911Edges deconstructor.
   allEdgesDeviceProps.countVertices_ = 0;
}


///  Copy all edge data from device to host.
///
///  @param  allEdgesDevice  GPU address of the All911EdgesDeviceProperties struct
///                             on device memory.
void All911Edges::copyEdgeDeviceToHost()
{
   // copy everything necessary
   All911EdgesDeviceProperties allEdgesDeviceProps;
   GPUModel *gpuModel = static_cast<GPUModel *>(&Simulator::getInstance().getModel());
   void *allEdgesDevice = static_cast<void *>(gpuModel->getAllEdgesDevice());
   HANDLE_ERROR(cudaMemcpy(&allEdgesDeviceProps, allEdgesDevice,
                           sizeof(All911EdgesDeviceProperties), cudaMemcpyDeviceToHost));
   copyDeviceToHost(allEdgesDeviceProps);
}

///  Copy all edge data from device to host.
///  (Helper function of copyEdgeDeviceToHost)
///
///  @param  allEdgesProperties     GPU address of the All911EdgesDeviceProperties struct
///                                on device memory.
void All911Edges::copyDeviceToHost(All911EdgesDeviceProperties &allEdgesDeviceProps)
{
   int numVertices = Simulator::getInstance().getTotalVertices();
   BGSIZE maxTotalEdges = Simulator::getInstance().getMaxEdgesPerVertex() * numVertices;
   HANDLE_ERROR(cudaMemcpy(sourceVertexIndex_.data(), allEdgesDeviceProps.sourceVertexIndex_,
                           maxTotalEdges * sizeof(int), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(destVertexIndex_.data(), allEdgesDeviceProps.destVertexIndex_,
                           maxTotalEdges * sizeof(int), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(W_.data(), allEdgesDeviceProps.W_, maxTotalEdges * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(type_.data(), allEdgesDeviceProps.type_,
                           maxTotalEdges * sizeof(edgeType), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(inUse_.data(), allEdgesDeviceProps.inUse_,
                           maxTotalEdges * sizeof(unsigned char), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(edgeCounts_.data(), allEdgesDeviceProps.edgeCounts_,
                           numVertices * sizeof(BGSIZE), cudaMemcpyDeviceToHost));
   totalEdgeCount_ = allEdgesDeviceProps.totalEdgeCount_;
   maxEdgesPerVertex_ = allEdgesDeviceProps.maxEdgesPerVertex_;
   countVertices_ = allEdgesDeviceProps.countVertices_;
   // Set countVertices_ to 0 to avoid illegal memory deallocation
   // at All911Edges deconstructor.
   allEdgesDeviceProps.countVertices_ = 0;
   HANDLE_ERROR(cudaMemcpy(isAvailable_.data(), allEdgesDeviceProps.isAvailable_,
                           maxTotalEdges * sizeof(unsigned char), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(isRedial_.data(), allEdgesDeviceProps.isRedial_,
                           maxTotalEdges * sizeof(unsigned char), cudaMemcpyDeviceToHost));

   // Bracket array declaration to memcpy to manually release array from stack
   // This is necessary to prevent segmentation faults when running large graphs
   {
      int cpuVertexId[maxTotalEdges];
      HANDLE_ERROR(cudaMemcpy(cpuVertexId, allEdgesDeviceProps.vertexId_,
                              maxTotalEdges * sizeof(int), cudaMemcpyDeviceToHost));
      for (int i = 0; i < maxTotalEdges; i++) {
         call_[i].vertexId = cpuVertexId[i];
      }
   }

   // Bracket array declaration to memcpy to manually release array from stack
   // This is necessary to prevent segmentation faults when running large graphs
   {
      uint64_t cpuTime[maxTotalEdges];
      HANDLE_ERROR(cudaMemcpy(cpuTime, allEdgesDeviceProps.time_, maxTotalEdges * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < maxTotalEdges; i++) {
         call_[i].time = cpuTime[i];
      }
   }

   // Bracket array declaration to memcpy to manually release array from stack
   // This is necessary to prevent segmentation faults when running large graphs
   {
      int cpuDuration[maxTotalEdges];
      HANDLE_ERROR(cudaMemcpy(cpuDuration, allEdgesDeviceProps.duration_,
                              maxTotalEdges * sizeof(int), cudaMemcpyDeviceToHost));
      for (int i = 0; i < maxTotalEdges; i++) {
         call_[i].duration = cpuDuration[i];
      }
   }

   // Bracket array declaration to memcpy to manually release array from stack
   // This is necessary to prevent segmentation faults when running large graphs
   {
      BGFLOAT cpuX[maxTotalEdges];
      HANDLE_ERROR(cudaMemcpy(cpuX, allEdgesDeviceProps.x_, maxTotalEdges * sizeof(BGFLOAT),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < maxTotalEdges; i++) {
         call_[i].x = cpuX[i];
      }
   }

   // Bracket array declaration to memcpy to manually release array from stack
   // This is necessary to prevent segmentation faults when running large graphs
   {
      BGFLOAT cpuY[maxTotalEdges];
      HANDLE_ERROR(cudaMemcpy(cpuY, allEdgesDeviceProps.y_, maxTotalEdges * sizeof(BGFLOAT),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < maxTotalEdges; i++) {
         call_[i].y = cpuY[i];
      }
   }

   // Bracket array declaration to memcpy to manually release array from stack
   // This is necessary to prevent segmentation faults when running large graphs
   {
      int cpuPatience[maxTotalEdges];
      HANDLE_ERROR(cudaMemcpy(cpuPatience, allEdgesDeviceProps.patience_,
                              maxTotalEdges * sizeof(int), cudaMemcpyDeviceToHost));
      for (int i = 0; i < maxTotalEdges; i++) {
         call_[i].patience = cpuPatience[i];
      }
   }

   // Bracket array declaration to memcpy to manually release array from stack
   // This is necessary to prevent segmentation faults when running large graphs
   {
      int cpuOnSiteTime[maxTotalEdges];
      HANDLE_ERROR(cudaMemcpy(cpuOnSiteTime, allEdgesDeviceProps.onSiteTime_,
                              maxTotalEdges * sizeof(int), cudaMemcpyDeviceToHost));
      for (int i = 0; i < maxTotalEdges; i++) {
         call_[i].onSiteTime = cpuOnSiteTime[i];
      }
   }

   // Bracket array declaration to memcpy to manually release array from stack
   // This is necessary to prevent segmentation faults when running large graphs
   {
      int cpuResponderType[maxTotalEdges];
      HANDLE_ERROR(cudaMemcpy(cpuResponderType, allEdgesDeviceProps.responderType_,
                              maxTotalEdges * sizeof(int), cudaMemcpyDeviceToHost));
      for (int i = 0; i < maxTotalEdges; i++) {
         if (cpuResponderType[i] == 7)
            call_[i].type = "Law";
         else if (cpuResponderType[i] == 5)
            call_[i].type = "EMS";
         else if (cpuResponderType[i] == 6)
            call_[i].type = "Fire";
      }
   }
}

///  Get edge_counts in AllEdges struct on device memory.
///
///  @param  allEdgesDevice  GPU address of the All911EdgesDeviceProperties struct
///                             on device memory.
void All911Edges::copyDeviceEdgeCountsToHost(void *allEdgesDevice)
{
   All911EdgesDeviceProperties allEdgesDeviceProps;
   int vertexCount = Simulator::getInstance().getTotalVertices();
   HANDLE_ERROR(cudaMemcpy(&allEdgesDeviceProps, allEdgesDevice,
                           sizeof(All911EdgesDeviceProperties), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(edgeCounts_.data(), allEdgesDeviceProps.edgeCounts_,
                           vertexCount * sizeof(BGSIZE), cudaMemcpyDeviceToHost));
}

///  Advance all the edges in the simulation.
///  Update the state of all edges for a time step.
///
///  @param  allEdgesDevice      GPU address of the AllEdgesDeviceProperties struct
///                                 on device memory.
///  @param  allVerticesDevice       GPU address of the AllVerticesDeviceProperties struct on device memory.
///  @param  edgeIndexMapDevice  GPU address of the EdgeIndexMap on device memory.
void All911Edges::advanceEdges(void *allEdgesDevice, void *allVerticesDevice,
                               void *edgeIndexMapDevice)
{
   //Edges are just used to store calls between vertices
}

void All911Edges::setAdvanceEdgesDeviceParams()
{
   //Advance edges does nothing so no params to set
}

///  Prints GPU 911EdgesProps data.
///
///  @param  allEdgesDeviceProps   GPU address of the corresponding All911EdgesDeviceProperties struct on device memory.
void All911Edges::printGPUEdgesProps(void *allEdgesDeviceProps) const
{
   All911EdgesDeviceProperties allEdgesProps;
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
      BGFLOAT *xPrint = new BGFLOAT[size];
      BGFLOAT *yPrint = new BGFLOAT[size];
      int *patiencePrint = new int[size];
      int *onSiteTimePrint = new int[size];
      int *responderTypePrint = new int[size];

      //set some array to default values
      //TODO: should look into why this is necessary
      for (BGSIZE i = 0; i < size; i++) {
         inUsePrint[i] = false;
      }

      for (int i = 0; i < countVertices_; i++) {
         edgeCountsPrint[i] = 0;
      }

      // copy everything
      HANDLE_ERROR(cudaMemcpy(&allEdgesProps, allEdgesDeviceProps,
                              sizeof(All911EdgesDeviceProperties), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(sourceVertexIndexPrint, allEdgesProps.sourceVertexIndex_,
                              size * sizeof(int), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(destVertexIndexPrint, allEdgesProps.destVertexIndex_,
                              size * sizeof(int), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(
         cudaMemcpy(WPrint, allEdgesProps.W_, size * sizeof(BGFLOAT), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(typePrint, allEdgesProps.type_, size * sizeof(edgeType),
                              cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(inUsePrint, allEdgesProps.inUse_, size * sizeof(unsigned char),
                              cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(edgeCountsPrint, allEdgesProps.edgeCounts_,
                              countVertices_ * sizeof(BGSIZE), cudaMemcpyDeviceToHost));
      totalEdgeCountPrint = allEdgesProps.totalEdgeCount_;
      maxEdgesPerVertexPrint = allEdgesProps.maxEdgesPerVertex_;
      countVerticesPrint = allEdgesProps.countVertices_;
      // Set countVertices_ to 0 to avoid illegal memory deallocation
      // at AllSynapsesProps deconstructor.
      allEdgesProps.countVertices_ = 0;
      HANDLE_ERROR(cudaMemcpy(isAvailablePrint, allEdgesProps.isAvailable_,
                              size * sizeof(unsigned char), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(isRedialPrint, allEdgesProps.isRedial_, size * sizeof(unsigned char),
                              cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(vertexIdPrint, allEdgesProps.vertexId_, size * sizeof(int),
                              cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(timePrint, allEdgesProps.time_, size * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(durationPrint, allEdgesProps.duration_, size * sizeof(int),
                              cudaMemcpyDeviceToHost));
      HANDLE_ERROR(
         cudaMemcpy(xPrint, allEdgesProps.x_, size * sizeof(BGFLOAT), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(
         cudaMemcpy(yPrint, allEdgesProps.y_, size * sizeof(BGFLOAT), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(patiencePrint, allEdgesProps.patience_, size * sizeof(int),
                              cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(onSiteTimePrint, allEdgesProps.onSiteTime_, size * sizeof(int),
                              cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(responderTypePrint, allEdgesProps.responderType_, size * sizeof(int),
                              cudaMemcpyDeviceToHost));

      //Print everything
      for (BGSIZE i = 0; i < size; i++) {
         if (WPrint[i] != 0.0) {
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