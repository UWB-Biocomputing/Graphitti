/**
 * @file AllIZHNeurons_d.cpp
 *
 * @brief A container of all Izhikevich neuron data
 *
 * @ingroup Simulator/Vertices
 */

#include "AllIZHNeurons.h"
#include "AllSpikingSynapses.h"
#include "AllVerticesDeviceFuncs.h"
#include "Book.h"
#include "Simulator.h"
#include "GPUModel.h"

///  CUDA code for advancing izhikevich neurons
///
///  @param[in] totalVertices          Number of vertices.
///  @param[in] maxEdges           Maximum number of synapses per neuron.
///  @param[in] maxSpikes             Maximum number of spikes per neuron per epoch.
///  @param[in] deltaT                Inner simulation step duration.
///  @param[in] simulationStep        The current simulation step.
///  @param[in] randNoise             Pointer to device random noise array.
///  @param[in] allVerticesDevice      Pointer to Neuron structures in device memory.
///  @param[in] allEdgesDevice     Pointer to Synapse structures in device memory.
///  @param[in] edgeIndexMap       Inverse map, which is a table indexed by an input neuron and maps to the synapses that provide input to that neuron.
///  @param[in] fAllowBackPropagation True if back propagaion is allowed.

__global__ void advanceIZHNeuronsDevice(int totalVertices, int maxEdges, int maxSpikes,
                                        BGFLOAT deltaT, uint64_t simulationStep, float randNoise[],
                                        AllIZHNeuronsDeviceProperties *allVerticesDevice,
                                        AllSpikingSynapsesDeviceProperties *allEdgesDevice,
                                        EdgeIndexMapDevice *edgeIndexMapDevice,
                                        bool fAllowBackPropagation);


///  Allocate GPU memories to store all neurons' states,
///  and copy them from host to GPU memory.
///
void AllIZHNeurons::allocNeuronDeviceStruct()
{
   AllIZHNeuronsDeviceProperties allVerticesDeviceProps;
   GPUModel* gpuModel = static_cast<GPUModel*>(&Simulator::getInstance().getModel());
   void** allVerticesDevice = reinterpret_cast<void**>(&(gpuModel->getAllVerticesDevice()));
   allocDeviceStruct(allVerticesDeviceProps);

   HANDLE_ERROR(cudaMalloc(allVerticesDevice, sizeof(AllIZHNeuronsDeviceProperties)));
   HANDLE_ERROR(cudaMemcpy(*allVerticesDevice, &allVerticesDeviceProps,
                           sizeof(AllIZHNeuronsDeviceProperties), cudaMemcpyHostToDevice));
}

///  Allocate GPU memories to store all neurons' states.
///  (Helper function of allocNeuronDeviceStruct)
///
///  @param  allVerticesDevice    GPU address of the AllIZHNeuronsDeviceProperties struct on device memory.
void AllIZHNeurons::allocDeviceStruct(AllIZHNeuronsDeviceProperties &allVerticesDevice)
{
   int count = Simulator::getInstance().getTotalVertices();

   AllIFNeurons::allocDeviceStruct(allVerticesDevice);

   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.Aconst_, count * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.Bconst_, count * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.Cconst_, count * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.Dconst_, count * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.u_, count * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.C3_, count * sizeof(BGFLOAT)));
}

///  Delete GPU memories.
///
///  @param  allVerticesDevice   GPU address of the AllIZHNeuronsDeviceProperties struct
///                             on device memory.
void AllIZHNeurons::deleteNeuronDeviceStruct(void *allVerticesDevice)
{
   AllIZHNeuronsDeviceProperties allVerticesDeviceProps;

   HANDLE_ERROR(cudaMemcpy(&allVerticesDeviceProps, allVerticesDevice,
                           sizeof(AllIZHNeuronsDeviceProperties), cudaMemcpyDeviceToHost));

   deleteDeviceStruct(allVerticesDeviceProps);

   HANDLE_ERROR(cudaFree(allVerticesDevice));
}

///  Delete GPU memories.
///  (Helper function of deleteNeuronDeviceStruct)
///
///  @param  allVerticesDevice    GPU address of the AllIZHNeuronsDeviceProperties struct on device memory.
void AllIZHNeurons::deleteDeviceStruct(AllIZHNeuronsDeviceProperties &allVerticesDevice)
{
   HANDLE_ERROR(cudaFree(allVerticesDevice.Aconst_));
   HANDLE_ERROR(cudaFree(allVerticesDevice.Bconst_));
   HANDLE_ERROR(cudaFree(allVerticesDevice.Cconst_));
   HANDLE_ERROR(cudaFree(allVerticesDevice.Dconst_));
   HANDLE_ERROR(cudaFree(allVerticesDevice.u_));
   HANDLE_ERROR(cudaFree(allVerticesDevice.C3_));

   AllIFNeurons::deleteDeviceStruct(allVerticesDevice);
}

///  Copy all neurons' data from host to device.
///
void AllIZHNeurons::copyToDevice()
{
   AllIZHNeuronsDeviceProperties allVerticesDeviceProps;
   GPUModel* gpuModel = static_cast<GPUModel*>(&Simulator::getInstance().getModel());
   void* allVerticesDevice = static_cast<void*>(gpuModel->getAllVerticesDevice());
   HANDLE_ERROR(cudaMemcpy(&allVerticesDeviceProps, allVerticesDevice,
                           sizeof(AllIZHNeuronsDeviceProperties), cudaMemcpyDeviceToHost));

   int count = Simulator::getInstance().getTotalVertices();

   AllIFNeurons::copyToDevice();

   HANDLE_ERROR(cudaMemcpy(allVerticesDeviceProps.Aconst_, Aconst_.data(), count * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allVerticesDeviceProps.Bconst_, Bconst_.data(), count * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allVerticesDeviceProps.Cconst_, Cconst_.data(), count * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allVerticesDeviceProps.Dconst_, Dconst_.data(), count * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allVerticesDeviceProps.u_, u_.data(), count * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allVerticesDeviceProps.C3_, C3_.data(), count * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
}

///  Copy all neurons' data from device to host.
///
void AllIZHNeurons::copyFromDevice()
{
   GPUModel* gpuModel = static_cast<GPUModel*>(&Simulator::getInstance().getModel());
   void* allVerticesDevice = static_cast<void*>(gpuModel->getAllVerticesDevice());
   AllIFNeurons::copyFromDevice();
   AllIZHNeuronsDeviceProperties allVerticesDeviceProps;

   HANDLE_ERROR(cudaMemcpy(&allVerticesDeviceProps, allVerticesDevice,
                           sizeof(AllIZHNeuronsDeviceProperties), cudaMemcpyDeviceToHost));

   int count = Simulator::getInstance().getTotalVertices();

   HANDLE_ERROR(cudaMemcpy(Aconst_.data(), allVerticesDeviceProps.Aconst_, count * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(Bconst_.data(), allVerticesDeviceProps.Bconst_, count * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(Cconst_.data(), allVerticesDeviceProps.Cconst_, count * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(Dconst_.data(), allVerticesDeviceProps.Dconst_, count * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(u_.data(), allVerticesDeviceProps.u_, count * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(C3_.data(), allVerticesDeviceProps.C3_, count * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
}

///  Copy spike history data stored in device memory to host.
///
///  @param  allVerticesDevice   GPU address of the AllIZHNeuronsDeviceProperties struct
///                             on device memory.
// void AllIZHNeurons::copyNeuronDeviceSpikeHistoryToHost(void *allVerticesDevice)
// {
//    AllIZHNeuronsDeviceProperties allVerticesDeviceProps;
//    HANDLE_ERROR(cudaMemcpy(&allVerticesDeviceProps, allVerticesDevice,
//                            sizeof(AllIZHNeuronsDeviceProperties), cudaMemcpyDeviceToHost));
//    AllSpikingNeurons::copyDeviceSpikeHistoryToHost(allVerticesDeviceProps);
// }


///  Clear the spike counts out of all neurons.
///
///  @param  allVerticesDevice   GPU address of the AllIZHNeuronsDeviceProperties struct
///                             on device memory.
void AllIZHNeurons::clearNeuronSpikeCounts(void *allVerticesDevice)
{
   AllIZHNeuronsDeviceProperties allVerticesDeviceProps;
   HANDLE_ERROR(cudaMemcpy(&allVerticesDeviceProps, allVerticesDevice,
                           sizeof(AllIZHNeuronsDeviceProperties), cudaMemcpyDeviceToHost));
   AllSpikingNeurons::clearDeviceSpikeCounts(allVerticesDeviceProps);
}

///  Notify outgoing synapses if neuron has fired.
void AllIZHNeurons::advanceVertices(AllEdges &synapses, void *allVerticesDevice,
                                    void *allEdgesDevice, float randNoise[],
                                    EdgeIndexMapDevice *edgeIndexMapDevice)
{
   int vertex_count = Simulator::getInstance().getTotalVertices();
   int maxSpikes = (int)((Simulator::getInstance().getEpochDuration()
                          * Simulator::getInstance().getMaxFiringRate()));

   // CUDA parameters
   const int threadsPerBlock = 256;
   int blocksPerGrid = (vertex_count + threadsPerBlock - 1) / threadsPerBlock;

   // Advance neurons ------------->
   advanceIZHNeuronsDevice<<<blocksPerGrid, threadsPerBlock>>>(
      vertex_count, Simulator::getInstance().getMaxEdgesPerVertex(), maxSpikes,
      Simulator::getInstance().getDeltaT(), g_simulationStep, randNoise,
      (AllIZHNeuronsDeviceProperties *)allVerticesDevice,
      (AllSpikingSynapsesDeviceProperties *)allEdgesDevice, edgeIndexMapDevice,
      fAllowBackPropagation_);
}


///  CUDA code for advancing izhikevich neurons
///
///  @param[in] totalVertices          Number of vertices.
///  @param[in] maxEdges           Maximum number of synapses per neuron.
///  @param[in] maxSpikes             Maximum number of spikes per neuron per epoch.
///  @param[in] deltaT                Inner simulation step duration.
///  @param[in] simulationStep        The current simulation step.
///  @param[in] randNoise             Pointer to device random noise array.
///  @param[in] allVerticesDevice      Pointer to Neuron structures in device memory.
///  @param[in] allEdgesDevice     Pointer to Synapse structures in device memory.
///  @param[in] edgeIndexMap       Inverse map, which is a table indexed by an input neuron and maps to the synapses that provide input to that neuron.
///  @param[in] fAllowBackPropagation True if back propagaion is allowed.
__global__ void advanceIZHNeuronsDevice(int totalVertices, int maxEdges, int maxSpikes,
                                        BGFLOAT deltaT, uint64_t simulationStep, float randNoise[],
                                        AllIZHNeuronsDeviceProperties *allVerticesDevice,
                                        AllSpikingSynapsesDeviceProperties *allEdgesDevice,
                                        EdgeIndexMapDevice *edgeIndexMapDevice,
                                        bool fAllowBackPropagation)
{
   // determine which neuron this thread is processing
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx >= totalVertices)
      return;

   allVerticesDevice->hasFired_[idx] = false;
   BGFLOAT &sp = allVerticesDevice->summationPoints_[idx];
   BGFLOAT &vm = allVerticesDevice->Vm_[idx];
   BGFLOAT &a = allVerticesDevice->Aconst_[idx];
   BGFLOAT &b = allVerticesDevice->Bconst_[idx];
   BGFLOAT &u = allVerticesDevice->u_[idx];
   BGFLOAT r_sp = sp;
   BGFLOAT r_vm = vm;
   BGFLOAT r_a = a;
   BGFLOAT r_b = b;
   BGFLOAT r_u = u;

   if (allVerticesDevice->numStepsInRefractoryPeriod_[idx] > 0) {   // is neuron refractory?
      --allVerticesDevice->numStepsInRefractoryPeriod_[idx];
   } else if (r_vm >= allVerticesDevice->Vthresh_[idx]) {   // should it fire?
      int &spikeCount = allVerticesDevice->numElementsInEpoch_[idx];
      // Note that the neuron has fired!
      allVerticesDevice->hasFired_[idx] = true;
      // record spike time
      int &queueEnd = allVerticesDevice->bufferEnd_[idx];
      allVerticesDevice->spikeHistory_[idx][queueEnd] = simulationStep;
      spikeCount++;

      queueEnd = (queueEnd + 1) % maxSpikes;

      // calculate the number of steps in the absolute refractory period
      allVerticesDevice->numStepsInRefractoryPeriod_[idx]
         = static_cast<int>(allVerticesDevice->Trefract_[idx] / deltaT + 0.5);

      // reset to 'Vreset'
      vm = allVerticesDevice->Cconst_[idx] * 0.001;
      u = r_u + allVerticesDevice->Dconst_[idx];

      // notify outgoing synapses of spike
      BGSIZE synapseCounts = edgeIndexMapDevice->outgoingEdgeCount_[idx];
      if (synapseCounts != 0) {
         // get the index of where this neuron's list of synapses are
         BGSIZE beginIndex = edgeIndexMapDevice->outgoingEdgeBegin_[idx];
         // get the memory location of where that list begins
         BGSIZE *outgoingMapBegin = &(edgeIndexMapDevice->outgoingEdgeIndexMap_[beginIndex]);

         // for each synapse, let them know we have fired
         for (BGSIZE i = 0; i < synapseCounts; i++) {
            preSpikingSynapsesSpikeHitDevice(outgoingMapBegin[i], allEdgesDevice);
         }
      }

      // notify incomming synapses of spike
      synapseCounts = edgeIndexMapDevice->incomingEdgeCount_[idx];
      if (fAllowBackPropagation && synapseCounts != 0) {
         // get the index of where this neuron's list of synapses are
         BGSIZE beginIndex = edgeIndexMapDevice->incomingEdgeBegin_[idx];
         // get the memory location of where that list begins
         BGSIZE *incomingMapBegin = &(edgeIndexMapDevice->incomingEdgeIndexMap_[beginIndex]);

         // for each synapse, let them know we have fired
         switch (classSynapses_d) {
            case classAllSTDPSynapses:
            case classAllDynamicSTDPSynapses:
               for (BGSIZE i = 0; i < synapseCounts; i++) {
                  postSTDPSynapseSpikeHitDevice(
                     incomingMapBegin[i],
                     static_cast<AllSTDPSynapsesDeviceProperties *>(allEdgesDevice));
               }   // end for
               break;

            case classAllSpikingSynapses:
            case classAllDSSynapses:
               for (BGSIZE i = 0; i < synapseCounts; i++) {
                  postSpikingSynapsesSpikeHitDevice(incomingMapBegin[i], allEdgesDevice);
               }   // end for
               break;

            default:
               assert(false);
         }   // end switch
      }
   } else {
      r_sp += allVerticesDevice->I0_[idx];   // add IO

      // Random number alg. goes here
      r_sp += (randNoise[idx] * allVerticesDevice->Inoise_[idx]);   // add cheap noise

      BGFLOAT Vint = r_vm * 1000;

      // Izhikevich model integration step
      BGFLOAT Vb = Vint + allVerticesDevice->C3_[idx] * (0.04 * Vint * Vint + 5 * Vint + 140 - u);
      u = r_u + allVerticesDevice->C3_[idx] * r_a * (r_b * Vint - r_u);

      vm = Vb * 0.001 + allVerticesDevice->C2_[idx] * r_sp;   // add inputs
   }

   // clear synaptic input for next time step
   sp = 0;
}
///@}
