/**
 * @file AllLIFNeurons_d.cpp
 * 
 * @ingroup Simulator/Vertices
 *
 * @brief A container of all LIF neuron data
 */

#include "AllLIFNeurons.h"
#include "AllVerticesDeviceFuncs.h"
#include "Book.h"


///  CUDA code for advancing LIF neurons
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

__global__ void advanceLIFNeuronsDevice(int totalVertices, int maxEdges, int maxSpikes,
                                        const BGFLOAT deltaT, uint64_t simulationStep,
                                        float randNoise[],
                                        AllIFNeuronsDeviceProperties *allVerticesDevice,
                                        AllSpikingSynapsesDeviceProperties *allEdgesDevice,
                                        EdgeIndexMapDevice *edgeIndexMapDevice,
                                        bool fAllowBackPropagation);


///  Update the state of all neurons for a time step
///  Notify outgoing synapses if neuron has fired.
///
///  @param  synapses               Reference to the allEdges struct on host memory.
///  @param  allVerticesDevice       GPU address of the allVerticesDeviceProperties struct
///                                 on device memory.
///  @param  allEdgesDevice      GPU address of the allEdgesDeviceProperties struct
///                                 on device memory.
///  @param  randNoise              Reference to the random noise array.
///  @param  edgeIndexMapDevice  GPU address of the EdgeIndexMap on device memory.
void AllLIFNeurons::advanceVertices(AllEdges &synapses, void *allVerticesDevice,
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
   advanceLIFNeuronsDevice<<<blocksPerGrid, threadsPerBlock>>>(
      vertex_count, Simulator::getInstance().getMaxEdgesPerVertex(), maxSpikes,
      Simulator::getInstance().getDeltaT(), g_simulationStep, randNoise,
      (AllIFNeuronsDeviceProperties *)allVerticesDevice,
      (AllSpikingSynapsesDeviceProperties *)allEdgesDevice, edgeIndexMapDevice,
      fAllowBackPropagation_);
}

///@}

/******************************************
* Global Functions for advanceVertices
******************************************/
///@{

///  CUDA code for advancing LIF neurons
///
///  @param[in] totalVertices          Number of vertices.
///  @param[in] maxEdges           Maximum number of synapses per neuron.
///  @param[in] maxSpikes             Maximum number of spikes per neuron per epoch.
///  @param[in] deltaT                Inner simulation step duration.
///  @param[in] simulationStep        The current simulation step.
///  @param[in] randNoise             Pointer to de/vice random noise array.
///  @param[in] allVerticesDevice      Pointer to Neuron structures in device memory.
///  @param[in] allEdgesDevice     Pointer to Synapse structures in device memory.
///  @param[in] edgeIndexMap       Inverse map, which is a table indexed by an input neuron and maps to the synapses that provide input to that neuron.
///  @param[in] fAllowBackPropagation True if back propagaion is allowed.
__global__ void advanceLIFNeuronsDevice(int totalVertices, int maxEdges, int maxSpikes,
                                        const BGFLOAT deltaT, uint64_t simulationStep,
                                        float randNoise[],
                                        AllIFNeuronsDeviceProperties *allVerticesDevice,
                                        AllSpikingSynapsesDeviceProperties *allEdgesDevice,
                                        EdgeIndexMapDevice *edgeIndexMapDevice,
                                        bool fAllowBackPropagation)
{
   // determine which neuron this thread is processing
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx >= totalVertices)
      return;

   allVerticesDevice->hasFired_[idx] = false;
   BGFLOAT &sp = allVerticesDevice->summationMap_[idx];
   BGFLOAT &vm = allVerticesDevice->Vm_[idx];
   BGFLOAT r_sp = sp;
   BGFLOAT r_vm = vm;

   if (allVerticesDevice->numStepsInRefractoryPeriod_[idx] > 0) {   // is neuron refractory?
      --allVerticesDevice->numStepsInRefractoryPeriod_[idx];
   } else if (r_vm >= allVerticesDevice->Vthresh_[idx]) {   // should it fire?
      int &spikeCount = allVerticesDevice->numEventsInEpoch_[idx];

      // Note that the neuron has fired!
      allVerticesDevice->hasFired_[idx] = true;

      // record spike time
      int &queueEnd = allVerticesDevice->queueEnd_[idx];
      //int idxSp = allVerticesDevice->queueEnd_[idx];
      allVerticesDevice->spikeHistory_[idx][queueEnd] = simulationStep;
      spikeCount++;

      queueEnd = (queueEnd + 1) % maxSpikes;
      // Debug statements to be removed
      // DEBUG_SYNAPSE(
      //     printf("advanceLIFNeuronsDevice\n");
      //     printf("          index: %d\n", idx);
      //     printf("          simulationStep: %d\n\n", simulationStep);
      // );

      // calculate the number of steps in the absolute refractory period
      allVerticesDevice->numStepsInRefractoryPeriod_[idx]
         = static_cast<int>(allVerticesDevice->Trefract_[idx] / deltaT + 0.5);

      // reset to 'Vreset'
      vm = allVerticesDevice->Vreset_[idx];

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
      vm = allVerticesDevice->C1_[idx] * r_vm
           + allVerticesDevice->C2_[idx] * (r_sp);   // decay Vm and add inputs
   }

   // clear synaptic input for next time step
   sp = 0;
}
