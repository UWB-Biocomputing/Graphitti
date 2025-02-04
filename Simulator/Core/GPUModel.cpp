/**
 * @file GPUModel.cpp
 * 
 * @ingroup Simulator/Core
 *
 * @brief Implementation of Model for the spiking neural networks.
 * 
 */

#include "GPUModel.h"
#include "AllEdges.h"
#include "AllSynapsesDeviceFuncs.h"
#include "AllVertices.h"
#include "Connections.h"
#include "Global.h"

#ifdef PERFORMANCE_METRICS
float g_time;
cudaEvent_t start, stop;
#endif   // PERFORMANCE_METRICS

__constant__ int d_debug_mask[1];

GPUModel::GPUModel() :
   Model::Model(), edgeIndexMapDevice_(nullptr), randNoise_d(nullptr),
   allVerticesDevice_(nullptr), allEdgesDevice_(nullptr)
{
}

/// Allocates  and initializes memories on CUDA device.
/// @param[out] allVerticesDevice          Memory location of the pointer to the neurons list on device memory.
/// @param[out] allEdgesDevice         Memory location of the pointer to the synapses list on device memory.
void GPUModel::allocDeviceStruct(void **allVerticesDevice, void **allEdgesDevice)
{
   // Get neurons and synapses
   AllVertices &vertices = layout_->getVertices();
   AllEdges &edges = connections_->getEdges();

   // Allocate Neurons and Synapses structs on GPU device memory
   vertices.allocNeuronDeviceStruct(allVerticesDevice);
   edges.allocEdgeDeviceStruct(allEdgesDevice);

   // Allocate memory for random noise array
   int numVertices = Simulator::getInstance().getTotalVertices();
   BGSIZE randNoise_d_size = numVertices * sizeof(float);   // size of random noise array
   HANDLE_ERROR(cudaMalloc((void **)&randNoise_d, randNoise_d_size));

   // Copy host neuron and synapse arrays into GPU device
   vertices.copyToDevice(*allVerticesDevice);
   edges.copyEdgeHostToDevice(*allEdgesDevice);

   // Allocate synapse inverse map in device memory
   allocEdgeIndexMap(numVertices);
}

/// Copies device memories to host memories and deallocates them.
/// @param[out] allVerticesDevice          Memory location of the pointer to the neurons list on device memory.
/// @param[out] allEdgesDevice         Memory location of the pointer to the synapses list on device memory.
void GPUModel::deleteDeviceStruct(void **allVerticesDevice, void **allEdgesDevice)
{
   // Get neurons and synapses
   AllVertices &vertices = layout_->getVertices();
   AllEdges &edges = connections_->getEdges();

   // Copy device synapse and neuron structs to host memory
   vertices.copyFromDevice(*allVerticesDevice);
   // Deallocate device memory
   vertices.deleteNeuronDeviceStruct(*allVerticesDevice);
   // Copy device synapse and neuron structs to host memory
   edges.copyEdgeDeviceToHost(*allEdgesDevice);
   // Deallocate device memory
   edges.deleteEdgeDeviceStruct(*allEdgesDevice);
   HANDLE_ERROR(cudaFree(randNoise_d));
}

/// Sets up the Simulation.
void GPUModel::setupSim()
{
   // Set device ID
   HANDLE_ERROR(cudaSetDevice(g_deviceId));
   // Set DEBUG flag
   HANDLE_ERROR(cudaMemcpyToSymbol(d_debug_mask, &g_debug_mask, sizeof(int)));
   Model::setupSim();

   //initialize Mersenne Twister
   //assuming numVertices >= 100 and is a multiple of 100. Note rng_mt_rng_count must be <= MT_RNG_COUNT
   int rng_blocks = 25;   //# of blocks the kernel will use
   int rng_nPerRng
      = 4;   //# of iterations per thread (thread granularity, # of rands generated per thread)
   int rng_mt_rng_count = Simulator::getInstance().getTotalVertices()
                          / rng_nPerRng;   //# of threads to generate for numVertices rand #s
   int rng_threads = rng_mt_rng_count / rng_blocks;   //# threads per block needed
   initMTGPU(Simulator::getInstance().getNoiseRngSeed(), rng_blocks, rng_threads, rng_nPerRng,
             rng_mt_rng_count);

#ifdef PERFORMANCE_METRICS
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   t_gpu_rndGeneration = 0.0;
   t_gpu_advanceNeurons = 0.0;
   t_gpu_advanceSynapses = 0.0;
   t_gpu_calcSummation = 0.0;
#endif   // PERFORMANCE_METRICS

   // allocates memories on CUDA device
   allocDeviceStruct((void **)&allVerticesDevice_, (void **)&allEdgesDevice_);

   EdgeIndexMap &edgeIndexMap = connections_->getEdgeIndexMap();
   // copy inverse map to the device memory
   copyEdgeIndexMapHostToDevice(edgeIndexMap,
                                   Simulator::getInstance().getTotalVertices());

   AllEdges &edges = connections_->getEdges();
   // set some parameters used for advanceVerticesDevice
   layout_->getVertices().setAdvanceVerticesDeviceParams(edges);

   // set some parameters used for advanceEdgesDevice
   edges.setAdvanceEdgesDeviceParams();
}

/// Performs any finalization tasks on network following a simulation.
void GPUModel::finish()
{
   // deallocates memories on CUDA device
   deleteDeviceStruct((void **)&allVerticesDevice_, (void **)&allEdgesDevice_);
   deleteEdgeIndexMap();

#ifdef PERFORMANCE_METRICS
   cudaEventDestroy(start);
   cudaEventDestroy(stop);
#endif   // PERFORMANCE_METRICS
}

/// Advance everything in the model one time step. In this case, that
/// means calling all of the kernels that do the "micro step" updating
/// (i.e., NOT the stuff associated with growth).
void GPUModel::advance()
{
#ifdef PERFORMANCE_METRICS
   // Reset CUDA timer to start measurement of GPU operations
   cudaStartTimer();
#endif   // PERFORMANCE_METRICS

   // Get vertices and edges
   AllVertices &vertices = layout_->getVertices();
   AllEdges &edges = connections_->getEdges();

   normalMTGPU(randNoise_d);

#ifdef PERFORMANCE_METRICS
   cudaLapTime(t_gpu_rndGeneration);
   cudaStartTimer();
#endif   // PERFORMANCE_METRICS

   // display running info to console
   // Advance neurons ------------->
   vertices.advanceVertices(edges, allVerticesDevice_, allEdgesDevice_, randNoise_d,
                            edgeIndexMapDevice_);

#ifdef PERFORMANCE_METRICS
   cudaLapTime(t_gpu_advanceNeurons);
   cudaStartTimer();
#endif   // PERFORMANCE_METRICS

   // Advance synapses ------------->
   edges.advanceEdges(allEdgesDevice_, allVerticesDevice_, edgeIndexMapDevice_);

#ifdef PERFORMANCE_METRICS
   cudaLapTime(t_gpu_advanceSynapses);
   cudaStartTimer();
#endif   // PERFORMANCE_METRICS

   // integrate the inputs of the vertices
   vertices.integrateVertexInputs(allVerticesDevice_, edgeIndexMapDevice_, allEdgesDevice_);

#ifdef PERFORMANCE_METRICS
   cudaLapTime(t_gpu_calcSummation);
#endif   // PERFORMANCE_METRICS
}

/// Update the connection of all the Neurons and Synapses of the simulation.
void GPUModel::updateConnections()
{
   // Get neurons and synapses
   AllVertices &vertices = layout_->getVertices();
   AllEdges &edges = connections_->getEdges();

   dynamic_cast<AllSpikingNeurons &>(vertices).copyFromDevice(allVerticesDevice_);
   // dynamic_cast<AllSpikingNeurons *>(neurons.get())
   //    ->copyNeuronDeviceSpikeHistoryToHost(allVerticesDevice_);

   // Update Connections data
   if (connections_->updateConnections(vertices)) {
      connections_->updateSynapsesWeights(Simulator::getInstance().getTotalVertices(), vertices,
                                          edges, allVerticesDevice_, allEdgesDevice_,
                                          getLayout());
      // create synapse index map
      connections_->createEdgeIndexMap();
      // copy index map to the device memory
      copyEdgeIndexMapHostToDevice(connections_->getEdgeIndexMap(),
                                      Simulator::getInstance().getTotalVertices());
   }
}

/// Update the Neuron's history.
void GPUModel::updateHistory()
{
   Model::updateHistory();
   // clear spike count

   AllVertices &vertices = layout_->getVertices();
   dynamic_cast<AllSpikingNeurons &>(vertices).clearNeuronSpikeCounts(allVerticesDevice_);
}

/// Allocate device memory for synapse inverse map.
/// @param  count	The number of vertices.
void GPUModel::allocEdgeIndexMap(int count)
{
   EdgeIndexMapDevice edgeIndexMapDevice;

   HANDLE_ERROR(cudaMalloc((void **)&edgeIndexMapDevice.outgoingEdgeBegin_, count * sizeof(BGSIZE)));
   HANDLE_ERROR(cudaMalloc((void **)&edgeIndexMapDevice.outgoingEdgeCount_, count * sizeof(BGSIZE)));
   HANDLE_ERROR(cudaMemset(edgeIndexMapDevice.outgoingEdgeBegin_, 0, count * sizeof(BGSIZE)));
   HANDLE_ERROR(cudaMemset(edgeIndexMapDevice.outgoingEdgeCount_, 0, count * sizeof(BGSIZE)));
   HANDLE_ERROR(cudaMalloc((void **)&edgeIndexMapDevice.incomingEdgeBegin_, count * sizeof(BGSIZE)));
   HANDLE_ERROR(cudaMalloc((void **)&edgeIndexMapDevice.incomingEdgeCount_, count * sizeof(BGSIZE)));
   HANDLE_ERROR(cudaMemset(edgeIndexMapDevice.incomingEdgeBegin_, 0, count * sizeof(BGSIZE)));
   HANDLE_ERROR(cudaMemset(edgeIndexMapDevice.incomingEdgeCount_, 0, count * sizeof(BGSIZE)));
   HANDLE_ERROR(cudaMalloc((void **)&edgeIndexMapDevice_, sizeof(EdgeIndexMapDevice)));
   edgeIndexMapDevice.incomingEdgeIndexMap_ = nullptr;
   edgeIndexMapDevice.outgoingEdgeIndexMap_ = nullptr;
   HANDLE_ERROR(cudaMemcpy(edgeIndexMapDevice_, &edgeIndexMapDevice, sizeof(EdgeIndexMapDevice),
                           cudaMemcpyHostToDevice));
}

/// Deallocate device memory for synapse inverse map.
void GPUModel::deleteEdgeIndexMap()
{
   EdgeIndexMapDevice edgeIndexMapDevice;
   HANDLE_ERROR(cudaMemcpy(&edgeIndexMapDevice, edgeIndexMapDevice_, sizeof(EdgeIndexMapDevice),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaFree(edgeIndexMapDevice.outgoingEdgeBegin_));
   HANDLE_ERROR(cudaFree(edgeIndexMapDevice.outgoingEdgeCount_));
   HANDLE_ERROR(cudaFree(edgeIndexMapDevice.outgoingEdgeIndexMap_));
   HANDLE_ERROR(cudaFree(edgeIndexMapDevice.incomingEdgeBegin_));
   HANDLE_ERROR(cudaFree(edgeIndexMapDevice.incomingEdgeCount_));
   HANDLE_ERROR(cudaFree(edgeIndexMapDevice.incomingEdgeIndexMap_));
   HANDLE_ERROR(cudaFree(edgeIndexMapDevice_));
}

/// Copy EdgeIndexMap in host memory to EdgeIndexMap in device memory.
/// @param  edgeIndexMapHost		Reference to the EdgeIndexMap in host memory.
void GPUModel::copyEdgeIndexMapHostToDevice(EdgeIndexMap &edgeIndexMapHost, int numVertices)
{
   AllEdges &edges = connections_->getEdges();
   int totalEdgeCount = dynamic_cast<AllEdges &>(edges).totalEdgeCount_;
   if (totalEdgeCount == 0)
      return;
   EdgeIndexMapDevice edgeIndexMapDevice;
   HANDLE_ERROR(cudaMemcpy(&edgeIndexMapDevice, edgeIndexMapDevice_, sizeof(EdgeIndexMapDevice),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(edgeIndexMapDevice.outgoingEdgeBegin_,
                           edgeIndexMapHost.outgoingEdgeBegin_.data(),
                           numVertices * sizeof(BGSIZE), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(edgeIndexMapDevice.outgoingEdgeCount_,
                           edgeIndexMapHost.outgoingEdgeCount_.data(),
                           numVertices * sizeof(BGSIZE), cudaMemcpyHostToDevice));
   if (edgeIndexMapDevice.outgoingEdgeIndexMap_ != nullptr) {
      HANDLE_ERROR(cudaFree(edgeIndexMapDevice.outgoingEdgeIndexMap_));
   }
   HANDLE_ERROR(cudaMalloc((void **)&edgeIndexMapDevice.outgoingEdgeIndexMap_,
                           totalEdgeCount * sizeof(BGSIZE)));
   HANDLE_ERROR(cudaMemcpy(edgeIndexMapDevice.outgoingEdgeIndexMap_,
                           edgeIndexMapHost.outgoingEdgeIndexMap_.data(),
                           totalEdgeCount * sizeof(BGSIZE), cudaMemcpyHostToDevice));
   // active synapse map
   HANDLE_ERROR(cudaMemcpy(edgeIndexMapDevice.incomingEdgeBegin_,
                           edgeIndexMapHost.incomingEdgeBegin_.data(),
                           numVertices * sizeof(BGSIZE), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(edgeIndexMapDevice.incomingEdgeCount_,
                           edgeIndexMapHost.incomingEdgeCount_.data(),
                           numVertices * sizeof(BGSIZE), cudaMemcpyHostToDevice));
   // the number of synapses may change, so we reallocate the memory
   if (edgeIndexMapDevice.incomingEdgeIndexMap_ != nullptr) {
      HANDLE_ERROR(cudaFree(edgeIndexMapDevice.incomingEdgeIndexMap_));
      edgeIndexMapDevice.incomingEdgeIndexMap_ = nullptr;
   }
   HANDLE_ERROR(cudaMalloc((void **)&edgeIndexMapDevice.incomingEdgeIndexMap_,
                           totalEdgeCount * sizeof(BGSIZE)));
   HANDLE_ERROR(cudaMemcpy(edgeIndexMapDevice.incomingEdgeIndexMap_,
                           edgeIndexMapHost.incomingEdgeIndexMap_.data(),
                           totalEdgeCount * sizeof(BGSIZE), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(edgeIndexMapDevice_, &edgeIndexMapDevice, sizeof(EdgeIndexMapDevice),
                           cudaMemcpyHostToDevice));
}

/// Copy GPU Synapse data to CPU.
void GPUModel::copyGPUtoCPU()
{
   // copy device synapse structs to host memory
   connections_->getEdges().copyEdgeDeviceToHost(allEdgesDevice_);
}

/// Copy CPU Synapse data to GPU.
void GPUModel::copyCPUtoGPU()
{
   // copy host synapse structs to device memory
   connections_->getEdges().copyEdgeHostToDevice(allEdgesDevice_);
}

/// Print out SynapseProps on the GPU.
void GPUModel::printGPUEdgesPropsModel() const
{
   connections_->getEdges().printGPUEdgesProps(allEdgesDevice_);
}
