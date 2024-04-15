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

void AllSpikingNeurons::copyToDevice(void *deviceAddress)
{
   AllSpikingNeuronsDeviceProperties allVerticesDevice;
   HANDLE_ERROR(cudaMemcpy(&allVerticesDevice, deviceAddress,
                           sizeof(AllSpikingNeuronsDeviceProperties), cudaMemcpyDeviceToHost));

   int count = Simulator::getInstance().getTotalVertices();
   bool cpu_has_fired[count];
   for (int i = 0; i < count; i++) {
      cpu_has_fired[i] = hasFired_[i];
   }
   HANDLE_ERROR(cudaMemcpy(allVerticesDevice.hasFired_, cpu_has_fired, count * sizeof(bool),
                           cudaMemcpyHostToDevice));

   int cpu_spike_count[count];
   for (int i = 0; i < count; i++) {
      cpu_spike_count[i] = vertexEvents_[i].getNumEventsInEpoch();
   }
   HANDLE_ERROR(cudaMemcpy(allVerticesDevice.numEventsInEpoch_, cpu_spike_count,
                           count * sizeof(int), cudaMemcpyHostToDevice));

   int cpu_queue_front[count];
   for (int i = 0; i < count; i++) {
      cpu_queue_front[i] = vertexEvents_[i].queueFront_;
   }
   HANDLE_ERROR(cudaMemcpy(allVerticesDevice.queueFront_, cpu_queue_front, count * sizeof(int),
                           cudaMemcpyHostToDevice));

   int cpu_queue_end[count];
   for (int i = 0; i < count; i++) {
      cpu_queue_end[i] = vertexEvents_[i].queueEnd_;
   }
   HANDLE_ERROR(cudaMemcpy(allVerticesDevice.queueEnd_, cpu_queue_end, count * sizeof(int),
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
   int maxSpikes = vertexEvents_[0].dataSeries_.size() - 1;
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

   bool cpu_has_fired[numVertices];
   HANDLE_ERROR(cudaMemcpy(cpu_has_fired, allVerticesDevice.hasFired_, numVertices * sizeof(bool),
                           cudaMemcpyDeviceToHost));
   for (int i = 0; i < numVertices; i++) {
      hasFired_[i] = cpu_has_fired[i];
   }

   // We have to copy the whole state of the event buffer from GPU memory because
   // we reset it in CPU code and then copy the new state back to the GPU.
   int cpu_spike_count[numVertices];
   HANDLE_ERROR(cudaMemcpy(cpu_spike_count, allVerticesDevice.numEventsInEpoch_,
                           numVertices * sizeof(int), cudaMemcpyDeviceToHost));
   for (int i = 0; i < numVertices; i++) {
      vertexEvents_[i].numEventsInEpoch_ = cpu_spike_count[i];
   }

   int queue_front[numVertices];
   HANDLE_ERROR(cudaMemcpy(queue_front, allVerticesDevice.queueFront_, numVertices * sizeof(int),
                           cudaMemcpyDeviceToHost));
   for (int i = 0; i < numVertices; i++) {
      vertexEvents_[i].queueFront_ = queue_front[i];
   }

   int queue_end[numVertices];
   HANDLE_ERROR(cudaMemcpy(queue_end, allVerticesDevice.queueEnd_, numVertices * sizeof(int),
                           cudaMemcpyDeviceToHost));
   for (int i = 0; i < numVertices; i++) {
      vertexEvents_[i].queueEnd_ = queue_end[i];
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
   int maxSpikes = vertexEvents_[0].dataSeries_.size() - 1;
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
   int numVertices = Simulator::getInstance().getTotalVertices();

   HANDLE_ERROR(cudaMemset(allVerticesDevice.numEventsInEpoch_, 0, numVertices * sizeof(int)));

   vector<int> epochStart(numVertices);
   for (int i = 0; i < epochStart.size(); ++i) {
      epochStart[i] = vertexEvents_[i].queueEnd_;
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
