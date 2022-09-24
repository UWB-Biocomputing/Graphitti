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

void AllSpikingNeurons::copyToDevice( void * deviceAddress)
{

   AllSpikingNeuronsDeviceProperties allVerticesDevice;
   HANDLE_ERROR(cudaMemcpy(&allVerticesDevice, deviceAddress, sizeof(AllSpikingNeuronsDeviceProperties),
                           cudaMemcpyDeviceToHost));

   //AllSpikingNeuronsDeviceProperties *allVerticesDevice = (AllSpikingNeuronsDeviceProperties*)deviceAddress;
   int count = Simulator::getInstance().getTotalVertices();
   bool *cpu_has_fired;
   cpu_has_fired = nullptr;
   cpu_has_fired = new bool[count];
   for ( int i = 0; i < count; i++)
   {
      cpu_has_fired[i] = hasFired_[i];
   }
   HANDLE_ERROR(cudaMemcpy(allVerticesDevice.hasFired_, cpu_has_fired, count * sizeof(bool),
                           cudaMemcpyHostToDevice));
   delete[] cpu_has_fired;
  //SPikecount Good
   int *cpu_spike_count;
   cpu_spike_count = nullptr;
   cpu_spike_count = new int[count];
   for ( int i = 0; i < count; i++)
   {
      cpu_spike_count [i] = vertexEvents_[i].getNumEventsInEpoch();
   }
   HANDLE_ERROR(cudaMemcpy(allVerticesDevice.numEventsInEpoch_, cpu_spike_count, count * sizeof(int),
                           cudaMemcpyHostToDevice));
   delete[] cpu_spike_count;

   //QueueFront
  // int *cpu_queue_front;
   //cpu_queue_front = nullptr;
   //cpu_queue_front = new int[count];
   vector<int> cpu_queue_front(count);
   for ( int i = 0; i < count; i++)
   {
      cpu_queue_front [i] = vertexEvents_[i].queueFront_;
   }
   HANDLE_ERROR(cudaMemcpy(allVerticesDevice.queueFront_, cpu_queue_front.data(), count * sizeof(int),
                           cudaMemcpyHostToDevice));

   
   vector<int> cpu_queue_end(count);
   for ( int i = 0; i < count; i++)
   {
      cpu_queue_end [i] = vertexEvents_[i].queueEnd_;
   }
   HANDLE_ERROR(cudaMemcpy(allVerticesDevice.queueEnd_, cpu_queue_end.data(), count * sizeof(int),
                           cudaMemcpyHostToDevice));
   //delete[] cpu_queue_front;

   vector<int> cpu_queue_start(count);
   for ( int i = 0; i < count; i++)
   {
      cpu_queue_start [i] = vertexEvents_[i].epochStart_;
   }
   HANDLE_ERROR(cudaMemcpy(allVerticesDevice.epochStart_, cpu_queue_start.data(), count * sizeof(int),
                           cudaMemcpyHostToDevice));
   

     
   uint64_t *pSpikeHistory[count];
   HANDLE_ERROR(cudaMemcpy(pSpikeHistory, allVerticesDevice.spikeHistory_,
                           count * sizeof(uint64_t *), cudaMemcpyDeviceToHost));
   for (int i = 0; i < count; i++) {
      HANDLE_ERROR(cudaMemcpy(pSpikeHistory[i], vertexEvents_[i].eventTimeSteps_.data() , vertexEvents_[i].eventTimeSteps_.size() * sizeof(u_int64_t),
                              cudaMemcpyHostToDevice));
   } 
}


void AllSpikingNeurons::copyFromDevice(void * deviceAddress)
{
   int numVertices = Simulator::getInstance().getTotalVertices();
   // AllSpikingNeuronsDeviceProperties *allVerticesDevice = dynamic_cast<AllSpikingNeuronsDeviceProperties *>(deviceAddress);
   AllSpikingNeuronsDeviceProperties *allVerticesDevice = (AllSpikingNeuronsDeviceProperties*)deviceAddress;
   bool *cpu_has_fired;
   cpu_has_fired = nullptr;
   cpu_has_fired = new bool[numVertices];
   HANDLE_ERROR(cudaMemcpy(cpu_has_fired,  allVerticesDevice->hasFired_, numVertices * sizeof(bool),
                           cudaMemcpyDeviceToHost));
   for ( int i = 0; i < numVertices; i++)
   {
      hasFired_[i] = cpu_has_fired[i];
   }
   int *cpu_spike_count;
   cpu_spike_count = nullptr;
   cpu_spike_count = new int[numVertices];
    HANDLE_ERROR(cudaMemcpy(cpu_spike_count, allVerticesDevice->numEventsInEpoch_, numVertices * sizeof(int),
                           cudaMemcpyDeviceToHost));
   for ( int i = 0; i < numVertices; i++)
   {
      vertexEvents_[i].numEventsInEpoch_ = cpu_spike_count[i];
   }
   uint64_t *pSpikeHistory[numVertices];
   HANDLE_ERROR(cudaMemcpy(pSpikeHistory, allVerticesDevice->spikeHistory_,
                           numVertices * sizeof(uint64_t *), cudaMemcpyDeviceToHost));
   for (int i = 0; i < numVertices; i++) {
      HANDLE_ERROR(cudaMemcpy(vertexEvents_[i].eventTimeSteps_.data(), pSpikeHistory[i], vertexEvents_[i].eventTimeSteps_.size() * sizeof(u_int64_t),
                              cudaMemcpyDeviceToHost));
   }

   
}


///  Copy spike history data stored in device memory to host.
///
///  @param  allVerticesDevice   GPU address of the AllSpikingNeuronsDeviceProperties struct
///                             on device memory.
// void AllSpikingNeurons::copyDeviceSpikeHistoryToHost(
//    AllSpikingNeuronsDeviceProperties &allVerticesDevice)
// {
   
// }

// ///  Copy spike counts data stored in device memory to host.
// ///
// ///  @param  allVerticesDevice   GPU address of the AllSpikingNeuronsDeviceProperties struct
// ///                             on device memory.
// void AllSpikingNeurons::copyDeviceSpikeCountsToHost(
//    AllSpikingNeuronsDeviceProperties &allVerticesDevice)
// {
//    int numVertices = Simulator::getInstance().getTotalVertices();

//    HANDLE_ERROR(cudaMemcpy(spikeCount_, allVerticesDevice.spikeCount_, numVertices * sizeof(int),
//                            cudaMemcpyDeviceToHost));
//    }
// }

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
   for (int i = 0; i < epochStart.size(); ++i)
   {
      epochStart[i] = vertexEvents_[i].queueEnd_;
   }
   HANDLE_ERROR(cudaMemcpy(allVerticesDevice.epochStart_, epochStart.data(), numVertices *sizeof(int), cudaMemcpyHostToDevice ));

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
