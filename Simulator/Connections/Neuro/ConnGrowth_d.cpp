/**
 * @file ConnGrowth_d.cpp
 * 
 * @ingroup Simulator/Connections
 * 
 *
 * @brief Update the weights of the Synapses in the simulation.
 */

#include "AllSpikingNeurons.h"
#include "AllSpikingSynapses.h"
#include "AllSynapsesDeviceFuncs.h"
#include "Book.h"
#include "ConnGrowth.h"
#include "Simulator.h"

/*
 *  Update the weights of the Synapses in the simulation. To be clear,
 *  iterates through all source and destination vertices and updates their
 *  edge strengths from the weight matrix.
 *  Note: Platform Dependent.
 *
 *  @param  numVertices         The number of vertices to update.
 *  @param  vertices            The AllVertices object.
 *  @param  edges               The AllEdges object.
 *  @param  allVerticesDevice   GPU address to the AllVertices struct in device memory.
 *  @param  allEdgesDevice      GPU address to the AllEdges struct in device memory.
 *  @param  layout              The Layout object.
 *  @param  simulationStream    The cuda stream for all synchronous kernels.
 */
void ConnGrowth::updateEdgesWeights(int numVertices, AllVertices &vertices, AllEdges &edges,
                                    AllVerticesDeviceProperties *allVerticesDevice,
                                    AllEdgesDeviceProperties *allEdgesDevice, Layout &layout,
                                    cudaStream_t simulationStream)
{
   Simulator &simulator = Simulator::getInstance();
   // For now, we just set the weights to equal the areas. We will later
   // scale it and set its sign (when we index and get its sign).
   W_ = area_;

   BGFLOAT deltaT = simulator.getDeltaT();

   // CUDA parameters
   const int threadsPerBlock = 256;
   int blocksPerGrid;

   // allocate device memories
   BGSIZE W_d_size = simulator.getTotalVertices() * simulator.getTotalVertices() * sizeof(BGFLOAT);
   BGFLOAT *W_h = new BGFLOAT[W_d_size];
   BGFLOAT *W_d;
   HANDLE_ERROR(cudaMalloc((void **)&W_d, W_d_size));

   vertexType *neuronTypeMapD;
   HANDLE_ERROR(
      cudaMalloc((void **)&neuronTypeMapD, simulator.getTotalVertices() * sizeof(vertexType)));

   // copy weight data to the device memory
   for (int i = 0; i < simulator.getTotalVertices(); i++)
      for (int j = 0; j < simulator.getTotalVertices(); j++)
         W_h[i * simulator.getTotalVertices() + j] = W_(i, j);

   HANDLE_ERROR(cudaMemcpy(W_d, W_h, W_d_size, cudaMemcpyHostToDevice));

   HANDLE_ERROR(cudaMemcpy(neuronTypeMapD, layout.vertexTypeMap_.data(),
                           simulator.getTotalVertices() * sizeof(vertexType),
                           cudaMemcpyHostToDevice));

   blocksPerGrid = (simulator.getTotalVertices() + threadsPerBlock - 1) / threadsPerBlock;
   updateSynapsesWeightsDevice<<<blocksPerGrid, threadsPerBlock, 0, simulationStream>>>(
      simulator.getTotalVertices(), deltaT, W_d, simulator.getMaxEdgesPerVertex(),
      (AllSpikingNeuronsDeviceProperties *)allVerticesDevice,
      (AllSpikingSynapsesDeviceProperties *)allEdgesDevice, neuronTypeMapD);

   // free memories
   HANDLE_ERROR(cudaFree(W_d));
   delete[] W_h;

   HANDLE_ERROR(cudaFree(neuronTypeMapD));

   // copy device synapse count to host memory
   edges.copyDeviceEdgeCountsToHost(allEdgesDevice);
   // copy device synapse summation coordinate to host memory
   dynamic_cast<AllSpikingSynapses &>(edges).copyDeviceEdgeSumIdxToHost(allEdgesDevice);
}
