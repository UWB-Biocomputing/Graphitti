/*
 * @file AllSpikingNeurons_d.cpp
 * 
 * @ingroup Simulator/Vertices
 *
 * @brief A container of all spiking neuron data
 */

#include "AllSpikingNeurons.h"
#include "AllSpikingSynapses.h"
#include "Book.h"
#include "DeviceVector.h"

/// CUDA kernel for adding psr of all incoming synapses to summation points.
///
/// Calculate the sum of synaptic input to each neuron. One thread
/// corresponds to one neuron. Iterates sequentially through the
/// forward synapse index map (synapseIndexMapDevice_) to access only
/// existing synapses. Using this structure eliminates the need to skip
/// synapses that have undergone lazy deletion from the main
/// (allEdgesDevice) synapse structure. The forward map is
/// re-computed during each network restructure (once per epoch) to
/// ensure that all synapse pointers for a neuron are stored
/// contiguously.
///
/// @param[in] totalVertices           Number of vertices in the entire simulation.
/// @param[in,out] allVerticesDevice   Pointer to Neuron structures in device memory.
/// @param[in] synapseIndexMapDevice_  Pointer to forward map structures in device memory.
/// @param[in] allEdgesDevice      Pointer to Synapse structures in device memory.

__global__ void calcSummationPointDevice(int totalVertices, BGFLOAT *summationPoints_,
                                         EdgeIndexMapDevice *edgeIndexMapDevice,
                                         AllSpikingSynapsesDeviceProperties *allEdgesDevice);


void AllSpikingNeurons::copyToDevice(void *deviceAddress)
{
   AllSpikingNeuronsDeviceProperties allVerticesDevice;
   HANDLE_ERROR(cudaMemcpy(&allVerticesDevice, deviceAddress,
                           sizeof(AllSpikingNeuronsDeviceProperties), cudaMemcpyDeviceToHost));

   int count = Simulator::getInstance().getTotalVertices();

   //Device vector handles memory operations for hasFired_ and summationPoints_
   hasFired_.copyToDevice();
   summationPoints_.copyToDevice();

   //Handling memory operations for event buffer (device side) explicitly
   // since device vector does not support object types yet
   int cpu_spike_count[count];
   for (int i = 0; i < count; i++) {
      cpu_spike_count[i] = vertexEvents_[i].getNumElementsInEpoch();
   }
   HANDLE_ERROR(cudaMemcpy(allVerticesDevice.numElementsInEpoch_, cpu_spike_count,
                           count * sizeof(int), cudaMemcpyHostToDevice));

   int cpu_queue_front[count];
   for (int i = 0; i < count; i++) {
      cpu_queue_front[i] = vertexEvents_[i].bufferFront_;
   }
   HANDLE_ERROR(cudaMemcpy(allVerticesDevice.bufferFront_, cpu_queue_front, count * sizeof(int),
                           cudaMemcpyHostToDevice));

   int cpu_queue_end[count];
   for (int i = 0; i < count; i++) {
      cpu_queue_end[i] = vertexEvents_[i].bufferEnd_;
   }
   HANDLE_ERROR(cudaMemcpy(allVerticesDevice.bufferEnd_, cpu_queue_end, count * sizeof(int),
                           cudaMemcpyHostToDevice));
   int cpu_queue_start[count];
   for (int i = 0; i < count; i++) {
      cpu_queue_start[i] = vertexEvents_[i].epochStart_;
   }
   HANDLE_ERROR(cudaMemcpy(allVerticesDevice.epochStart_, cpu_queue_start, count * sizeof(int),
                           cudaMemcpyHostToDevice));

   uint64_t *pSpikeHistory[count];
   HANDLE_ERROR(cudaMemcpy(pSpikeHistory, allVerticesDevice.spikeHistory_,
                           count * sizeof(uint64_t *), cudaMemcpyDeviceToHost));

   // All EventBuffers are of the same size,
   // which is one greater than maxSpikes in GPU spikeHistory array.
   int maxSpikes = vertexEvents_[0].dataSeries_.size();
   for (int i = 0; i < count; i++) {
      HANDLE_ERROR(cudaMemcpy(pSpikeHistory[i], vertexEvents_[i].dataSeries_.data(),
                              maxSpikes * sizeof(uint64_t), cudaMemcpyHostToDevice));
   }
}
void AllSpikingNeurons::copyFromDevice(void *deviceAddress)
{
   int numVertices = Simulator::getInstance().getTotalVertices();

   AllSpikingNeuronsDeviceProperties allVerticesDevice;
   HANDLE_ERROR(cudaMemcpy(&allVerticesDevice, deviceAddress,
                           sizeof(AllSpikingNeuronsDeviceProperties), cudaMemcpyDeviceToHost));

   //Device vector handles memory operations for hasFired_ and summationPoints_
   hasFired_.copyToHost();
   summationPoints_.copyToHost();

   //Handling memory operations for event buffer (device side) explicitly
   // since device vector does not support object types yet

   // We have to copy the whole state of the event buffer from GPU memory because
   // we reset it in CPU code and then copy the new state back to the GPU.
   int cpu_spike_count[numVertices];
   HANDLE_ERROR(cudaMemcpy(cpu_spike_count, allVerticesDevice.numElementsInEpoch_,
                           numVertices * sizeof(int), cudaMemcpyDeviceToHost));
   for (int i = 0; i < numVertices; i++) {
      vertexEvents_[i].numElementsInEpoch_ = cpu_spike_count[i];
   }

   int queue_front[numVertices];
   HANDLE_ERROR(cudaMemcpy(queue_front, allVerticesDevice.bufferFront_, numVertices * sizeof(int),
                           cudaMemcpyDeviceToHost));
   for (int i = 0; i < numVertices; i++) {
      vertexEvents_[i].bufferFront_ = queue_front[i];
   }

   int queue_end[numVertices];
   HANDLE_ERROR(cudaMemcpy(queue_end, allVerticesDevice.bufferEnd_, numVertices * sizeof(int),
                           cudaMemcpyDeviceToHost));
   for (int i = 0; i < numVertices; i++) {
      vertexEvents_[i].bufferEnd_ = queue_end[i];
   }

   int epoch_start[numVertices];
   HANDLE_ERROR(cudaMemcpy(epoch_start, allVerticesDevice.epochStart_, numVertices * sizeof(int),
                           cudaMemcpyDeviceToHost));
   for (int i = 0; i < numVertices; i++) {
      vertexEvents_[i].epochStart_ = epoch_start[i];
   }

   uint64_t *pSpikeHistory[numVertices];
   HANDLE_ERROR(cudaMemcpy(pSpikeHistory, allVerticesDevice.spikeHistory_,
                           numVertices * sizeof(uint64_t), cudaMemcpyDeviceToHost));

   // All EventBuffers are of the same size,
   // which is one greater than maxSpikes in GPU spikeHistory array.
   int maxSpikes = vertexEvents_[0].dataSeries_.size();
   for (int i = 0; i < numVertices; i++) {
      HANDLE_ERROR(cudaMemcpy(vertexEvents_[i].dataSeries_.data(), pSpikeHistory[i],
                              maxSpikes * sizeof(uint64_t *), cudaMemcpyDeviceToHost));
   }
}

///  Clear the spike counts out of all neurons in device memory.
///  (helper function of clearNeuronSpikeCounts)
///
///  @param  allVerticesDevice   GPU address of the AllSpikingNeuronsDeviceProperties struct
///                             on device memory.
void AllSpikingNeurons::clearDeviceSpikeCounts(AllSpikingNeuronsDeviceProperties &allVerticesDevice)
{
   //Handling memory operations for event buffer (device side) explicitly
   // since device vector does not support object types yet

   int numVertices = Simulator::getInstance().getTotalVertices();

   HANDLE_ERROR(cudaMemset(allVerticesDevice.numElementsInEpoch_, 0, numVertices * sizeof(int)));

   vector<int> epochStart(numVertices);
   for (int i = 0; i < epochStart.size(); ++i) {
      epochStart[i] = vertexEvents_[i].bufferEnd_;
   }
   HANDLE_ERROR(cudaMemcpy(allVerticesDevice.epochStart_, epochStart.data(),
                           numVertices * sizeof(int), cudaMemcpyHostToDevice));
}

///  Set some parameters used for advanceVerticesDevice.
///  Currently we set the two member variables: m_fpPreSpikeHit_h and m_fpPostSpikeHit_h.
///  These are function pointers for PreSpikeHit and PostSpikeHit device functions
///  respectively, and these functions are called from advanceVerticesDevice device
///  function. We use this scheme because we cannot not use virtual function (Polymorphism)
///  in device functions.
///
///  @param  synapses               Reference to the allEdges struct on host memory.
void AllSpikingNeurons::setAdvanceVerticesDeviceParams(AllEdges &synapses)
{
   AllSpikingSynapses &spSynapses = dynamic_cast<AllSpikingSynapses &>(synapses);
   fAllowBackPropagation_ = spSynapses.allowBackPropagation();
}

/// Add psr of all incoming synapses to summation points.
///
/// @param allVerticesDevice       GPU address of the allVertices struct on device memory.
/// @param edgeIndexMapDevice      GPU address of the EdgeIndexMap on device memory.
/// @param allEdgesDevice          GPU address of the allEdges struct on device memory.
void AllSpikingNeurons::integrateVertexInputs(void *allVerticesDevice,
                                              EdgeIndexMapDevice *edgeIndexMapDevice,
                                              void *allEdgesDevice)
{
   // CUDA parameters
   const int threadsPerBlock = 256;
   int blocksPerGrid
      = (Simulator::getInstance().getTotalVertices() + threadsPerBlock - 1) / threadsPerBlock;
   int vertex_count = Simulator::getInstance().getTotalVertices();

   calcSummationPointDevice<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
      vertex_count, summationPoints_, edgeIndexMapDevice,
      (AllSpikingSynapsesDeviceProperties *)allEdgesDevice);
}

/// CUDA kernel for adding psr of all incoming synapses to summation points.
///
/// Calculate the sum of synaptic input to each neuron. One thread
/// corresponds to one neuron. Iterates sequentially through the
/// forward synapse index map (synapseIndexMapDevice_) to access only
/// existing synapses. Using this structure eliminates the need to skip
/// synapses that have undergone lazy deletion from the main
/// (allEdgesDevice) synapse structure. The forward map is
/// re-computed during each network restructure (once per epoch) to
/// ensure that all synapse pointers for a neuron are stored
/// contiguously.
///
/// @param[in] totalVertices           Number of vertices in the entire simulation.
/// @param[in,out] allVerticesDevice   Pointer to Neuron structures in device memory.
/// @param[in] edgeIndexMapDevice  Pointer to forward map structures in device memory.
/// @param[in] allEdgesDevice      Pointer to Synapse structures in device memory.

__global__ void calcSummationPointDevice(int totalVertices, BGFLOAT *summationPoints_,
                                         EdgeIndexMapDevice *edgeIndexMapDevice,
                                         AllSpikingSynapsesDeviceProperties *allEdgesDevice)
{
   // The usual thread ID calculation and guard against excess threads
   // (beyond the number of vertices, in this case).
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx >= totalVertices)
      return;

   // Number of incoming synapses
   BGSIZE synCount = edgeIndexMapDevice->incomingEdgeCount_[idx];
   // Optimization: terminate thread if no incoming synapses
   if (synCount != 0) {
      // Index of start of this neuron's block of forward map entries
      int beginIndex = edgeIndexMapDevice->incomingEdgeBegin_[idx];
      // Address of the start of this neuron's block of forward map entries
      BGSIZE *activeMapBegin = &(edgeIndexMapDevice->incomingEdgeIndexMap_[beginIndex]);
      // Summed post-synaptic response (PSR)
      BGFLOAT sum = 0.0;
      // Index of the current incoming synapse
      BGSIZE synIndex;
      // Repeat for each incoming synapse
      for (BGSIZE i = 0; i < synCount; i++) {
         // Get index of current incoming synapse
         synIndex = activeMapBegin[i];
         // Fetch its PSR and add into sum
         sum += allEdgesDevice->psr_[synIndex];
      }
      // Store summed PSR into this neuron's summation point
      summationPoints_[idx] = sum;
   }
}
