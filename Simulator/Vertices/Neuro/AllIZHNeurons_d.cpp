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
#include "DeviceVector.h"


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

__global__ void advanceIZHNeuronsDevice(
   int totalVertices, int maxEdges, int maxSpikes, BGFLOAT deltaT, uint64_t simulationStep,
   float randNoise[], bool *hasFired_, BGFLOAT *summationPoints_, BGFLOAT *Vm_, BGFLOAT *Aconst_,
   BGFLOAT *Bconst_, BGFLOAT *u_, int *numStepsInRefractoryPeriod_, BGFLOAT *Vthresh_,
   BGFLOAT *Trefract_, BGFLOAT *Cconst_, BGFLOAT *Dconst_, BGFLOAT *I0_, BGFLOAT *Inoise_,
   BGFLOAT *C3_, BGFLOAT *C2_, AllIZHNeuronsDeviceProperties *allVerticesDevice,
   AllSpikingSynapsesDeviceProperties *allEdgesDevice, EdgeIndexMapDevice *edgeIndexMapDevice,
   bool fAllowBackPropagation);


///  Allocate GPU memories to store all neurons' states,
///  and copy them from host to GPU memory.
///
///  @param  allVerticesDevice   GPU address of the AllIZHNeuronsDeviceProperties struct
///                             on device memory.
void AllIZHNeurons::allocVerticesDeviceStruct(void **allVerticesDevice)
{
   AllIZHNeuronsDeviceProperties allVerticesDeviceProps;

   allocDeviceStruct(allVerticesDeviceProps);

   HANDLE_ERROR(cudaMalloc(allVerticesDevice, sizeof(AllIZHNeuronsDeviceProperties)));
   HANDLE_ERROR(cudaMemcpy(*allVerticesDevice, &allVerticesDeviceProps,
                           sizeof(AllIZHNeuronsDeviceProperties), cudaMemcpyHostToDevice));
}

///  Allocate GPU memories to store all neurons' states.
///  (Helper function of allocVerticesDeviceStruct)
///
///  @param  allVerticesDevice    GPU address of the AllIZHNeuronsDeviceProperties struct on device memory.
void AllIZHNeurons::allocDeviceStruct(AllIZHNeuronsDeviceProperties &allVerticesDevice)
{
   AllIFNeurons::allocDeviceStruct(allVerticesDevice);

   Aconst_.allocateDeviceMemory();
   Bconst_.allocateDeviceMemory();
   Cconst_.allocateDeviceMemory();
   Dconst_.allocateDeviceMemory();
   u_.allocateDeviceMemory();
   C3_.allocateDeviceMemory();
}

///  Delete GPU memories.
///
///  @param  allVerticesDevice   GPU address of the AllVerticesDeviceProperties struct
///                             on device memory.
void AllIZHNeurons::deleteVerticesDeviceStruct(void *allVerticesDevice)
{
   AllIZHNeuronsDeviceProperties allVerticesDeviceProps;

   HANDLE_ERROR(cudaMemcpy(&allVerticesDeviceProps, allVerticesDevice,
                           sizeof(AllIZHNeuronsDeviceProperties), cudaMemcpyDeviceToHost));

   deleteDeviceStruct(allVerticesDeviceProps);

   HANDLE_ERROR(cudaFree(allVerticesDevice));
}

///  Delete GPU memories.
///  (Helper function of deleteVerticesDeviceStruct)
///
///  @param  allVerticesDevice    GPU address of the AllIZHNeuronsDeviceProperties struct on device memory.
void AllIZHNeurons::deleteDeviceStruct(AllIZHNeuronsDeviceProperties &allVerticesDevice)
{
   Aconst_.freeDeviceMemory();
   Bconst_.freeDeviceMemory();
   Cconst_.freeDeviceMemory();
   Dconst_.freeDeviceMemory();
   u_.freeDeviceMemory();
   C3_.freeDeviceMemory();

   AllIFNeurons::deleteDeviceStruct(allVerticesDevice);
}

///  Copy all neurons' data from host to device.
///
///  @param  allVerticesDevice   GPU address of the AllIZHNeuronsDeviceProperties struct
///                             on device memory.
void AllIZHNeurons::copyToDevice(void *allVerticesDevice)
{
   AllIFNeurons::copyToDevice(allVerticesDevice);

   Aconst_.copyToDevice();
   Bconst_.copyToDevice();
   Cconst_.copyToDevice();
   Dconst_.copyToDevice();
   u_.copyToDevice();
   C3_.copyToDevice();
}

///  Copy all neurons' data from device to host.
///
///  @param  allVerticesDevice   GPU address of the AllIZHNeuronsDeviceProperties struct
///                             on device memory.
void AllIZHNeurons::copyFromDevice(void *allVerticesDevice)
{
   AllIFNeurons::copyFromDevice(allVerticesDevice);

   Aconst_.copyToHost();
   Bconst_.copyToHost();
   Cconst_.copyToHost();
   Dconst_.copyToHost();
   u_.copyToHost();
   C3_.copyToHost();
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
void AllIZHNeurons::clearVertexHistory(void *allVerticesDevice)
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
   advanceIZHNeuronsDevice<<<blocksPerGrid, threadsPerBlock,0,stream>>>(
      vertex_count, Simulator::getInstance().getMaxEdgesPerVertex(), maxSpikes,
      Simulator::getInstance().getDeltaT(), g_simulationStep, randNoise, hasFired_,
      summationPoints_, Vm_, Aconst_, Bconst_, u_, numStepsInRefractoryPeriod_, Vthresh_, Trefract_,
      Cconst_, Dconst_, I0_, Inoise_, C3_, C2_, (AllIZHNeuronsDeviceProperties *)allVerticesDevice,
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
__global__ void advanceIZHNeuronsDevice(
   int totalVertices, int maxEdges, int maxSpikes, BGFLOAT deltaT, uint64_t simulationStep,
   float randNoise[], bool *hasFired_, BGFLOAT *summationPoints_, BGFLOAT *Vm_, BGFLOAT *Aconst_,
   BGFLOAT *Bconst_, BGFLOAT *u_, int *numStepsInRefractoryPeriod_, BGFLOAT *Vthresh_,
   BGFLOAT *Trefract_, BGFLOAT *Cconst_, BGFLOAT *Dconst_, BGFLOAT *I0_, BGFLOAT *Inoise_,
   BGFLOAT *C3_, BGFLOAT *C2_, AllIZHNeuronsDeviceProperties *allVerticesDevice,
   AllSpikingSynapsesDeviceProperties *allEdgesDevice, EdgeIndexMapDevice *edgeIndexMapDevice,
   bool fAllowBackPropagation)
{
   // determine which neuron this thread is processing
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx >= totalVertices)
      return;

   hasFired_[idx] = false;
   BGFLOAT &sp = summationPoints_[idx];
   BGFLOAT &vm = Vm_[idx];
   BGFLOAT &a = Aconst_[idx];
   BGFLOAT &b = Bconst_[idx];
   BGFLOAT &u = u_[idx];
   BGFLOAT r_sp = sp;
   BGFLOAT r_vm = vm;
   BGFLOAT r_a = a;
   BGFLOAT r_b = b;
   BGFLOAT r_u = u;

   if (numStepsInRefractoryPeriod_[idx] > 0) {   // is neuron refractory?
      --numStepsInRefractoryPeriod_[idx];
   } else if (r_vm >= Vthresh_[idx]) {   // should it fire?
      int &spikeCount = allVerticesDevice->numElementsInEpoch_[idx];
      // Note that the neuron has fired!
      hasFired_[idx] = true;
      // record spike time
      int &queueEnd = allVerticesDevice->bufferEnd_[idx];
      allVerticesDevice->spikeHistory_[idx][queueEnd] = simulationStep;
      spikeCount++;

      queueEnd = (queueEnd + 1) % maxSpikes;

      // calculate the number of steps in the absolute refractory period
      numStepsInRefractoryPeriod_[idx] = static_cast<int>(Trefract_[idx] / deltaT + 0.5);

      // reset to 'Vreset'
      vm = Cconst_[idx] * 0.001;
      u = r_u + Dconst_[idx];

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
            case enumClassSynapses::classAllSTDPSynapses:
            case enumClassSynapses::classAllDynamicSTDPSynapses:
               for (BGSIZE i = 0; i < synapseCounts; i++) {
                  postSTDPSynapseSpikeHitDevice(
                     incomingMapBegin[i],
                     static_cast<AllSTDPSynapsesDeviceProperties *>(allEdgesDevice));
               }   // end for
               break;

            case enumClassSynapses::classAllSpikingSynapses:
            case enumClassSynapses::classAllDSSynapses:
               for (BGSIZE i = 0; i < synapseCounts; i++) {
                  postSpikingSynapsesSpikeHitDevice(incomingMapBegin[i], allEdgesDevice);
               }   // end for
               break;

            default:
               assert(false);
         }   // end switch
      }
   } else {
      r_sp += I0_[idx];   // add IO

      // Random number alg. goes here
      r_sp += (randNoise[idx] * Inoise_[idx]);   // add cheap noise

      BGFLOAT Vint = r_vm * 1000;

      // Izhikevich model integration step
      BGFLOAT Vb = Vint + C3_[idx] * (0.04 * Vint * Vint + 5 * Vint + 140 - u);
      u = r_u + C3_[idx] * r_a * (r_b * Vint - r_u);

      vm = Vb * 0.001 + C2_[idx] * r_sp;   // add inputs
   }

   // clear synaptic input for next time step
   sp = 0;
}
///@}
