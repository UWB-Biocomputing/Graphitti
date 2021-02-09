/**
 * @file AllVerticesDeviceFuncs_d.cu
 * 
 * @ingroup Simulator/Vertices
 *
 * @brief
 */

#include "AllVerticesDeviceFuncs.h"
#include "AllSynapsesDeviceFuncs.h"

/******************************************
* Device Functions for advanceVertices
******************************************/
///@{

///  Prepares Synapse for a spike hit.
///
///  @param[in] iEdg                  Index of the Synapse to update.
///  @param[in] allSynapsesDevice     Pointer to AllSpikingSynapsesDeviceProperties structures 
///                                   on device memory.
__device__ void preSpikingSynapsesSpikeHitDevice( const BGSIZE iEdg, AllSpikingSynapsesDeviceProperties* allSynapsesDevice ) {
        uint32_t &delay_queue = allSynapsesDevice->delayQueue_[iEdg]; 
        int delayIdx = allSynapsesDevice->delayIndex_[iEdg];
        int ldelayQueue = allSynapsesDevice->delayQueueLength_[iEdg];
        int total_delay = allSynapsesDevice->totalDelay_[iEdg];

        // Add to spike queue

        // calculate index where to insert the spike into delayQueue
        int idx = delayIdx +  total_delay;
        if ( idx >= ldelayQueue ) {
                idx -= ldelayQueue;
        }

        // set a spike
        //assert( !(delay_queue[0] & (0x1 << idx)) );
        delay_queue |= (0x1 << idx);
}

///  Prepares Synapse for a spike hit (for back propagation).
///
///  @param[in] iEdg                  Index of the Synapse to update.
///  @param[in] allSynapsesDevice     Pointer to AllSpikingSynapsesDeviceProperties structures 
///                                   on device memory.
__device__ void postSpikingSynapsesSpikeHitDevice( const BGSIZE iEdg, AllSpikingSynapsesDeviceProperties* allSynapsesDevice ) {
}

///  Prepares Synapse for a spike hit (for back propagation).
///
///  @param[in] iEdg                  Index of the Synapse to update.
///  @param[in] allSynapsesDevice     Pointer to AllSTDPSynapsesDeviceProperties structures 
///                                   on device memory.
__device__ void postSTDPSynapseSpikeHitDevice( const BGSIZE iEdg, AllSTDPSynapsesDeviceProperties* allSynapsesDevice ) {
        uint32_t &delayQueue = allSynapsesDevice->delayQueuePost_[iEdg];
        int delayIndex = allSynapsesDevice->delayIndexPost_[iEdg];
        int delayQueueLength = allSynapsesDevice->delayQueuePostLength_[iEdg];
        int totalDelay = allSynapsesDevice->totalDelayPost_[iEdg];

        // Add to spike queue

        // calculate index where to insert the spike into delayQueue
        int idx = delayIndex +  totalDelay;
        if ( idx >= delayQueueLength ) {
                idx -= delayQueueLength;
        }

        // set a spike
        //assert( !(delay_queue[0] & (0x1 << idx)) );
        delayQueue |= (0x1 << idx);
}
///@}

/******************************************
* Global Functions for advanceVertices
******************************************/
///@{

///  CUDA code for advancing LIF neurons
/// 
///  @param[in] totalVertices          Number of neurons.
///  @param[in] maxSynapses           Maximum number of synapses per neuron.
///  @param[in] maxSpikes             Maximum number of spikes per neuron per epoch.
///  @param[in] deltaT                Inner simulation step duration.
///  @param[in] simulationStep        The current simulation step.
///  @param[in] randNoise             Pointer to de/vice random noise array.
///  @param[in] allNeuronsDevice      Pointer to Neuron structures in device memory.
///  @param[in] allSynapsesDevice     Pointer to Synapse structures in device memory.
///  @param[in] edgeIndexMap       Inverse map, which is a table indexed by an input neuron and maps to the synapses that provide input to that neuron.
///  @param[in] fAllowBackPropagation True if back propagaion is allowed.
__global__ void advanceLIFNeuronsDevice( int totalVertices, int maxSynapses, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, float* randNoise, AllIFNeuronsDeviceProperties* allNeuronsDevice, AllSpikingSynapsesDeviceProperties* allSynapsesDevice, EdgeIndexMap* synapseIndexMapDevice, bool fAllowBackPropagation ) {
        // determine which neuron this thread is processing
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx >= totalVertices )
                return;

        allNeuronsDevice->hasFired_[idx] = false;
        BGFLOAT& sp = allNeuronsDevice->summationMap_[idx];
        BGFLOAT& vm = allNeuronsDevice->Vm_[idx];
        BGFLOAT r_sp = sp;
        BGFLOAT r_vm = vm;

        if ( allNeuronsDevice->numStepsInRefractoryPeriod_[idx] > 0 ) { // is neuron refractory?
                --allNeuronsDevice->numStepsInRefractoryPeriod_[idx];
        } else if ( r_vm >= allNeuronsDevice->Vthresh_[idx] ) { // should it fire?
                int& spikeCount = allNeuronsDevice->spikeCount_[idx];
                int& spikeCountOffset = allNeuronsDevice->spikeCountOffset_[idx];

                // Note that the neuron has fired!
                allNeuronsDevice->hasFired_[idx] = true;

                // record spike time
                int idxSp = (spikeCount + spikeCountOffset) % maxSpikes;
                allNeuronsDevice->spikeHistory_[idx][idxSp] = simulationStep;
                spikeCount++;

                // Debug statements to be removed 
                // DEBUG_SYNAPSE(
                //     printf("advanceLIFNeuronsDevice\n");
                //     printf("          index: %d\n", idx);
                //     printf("          simulationStep: %d\n\n", simulationStep);
                // );

                // calculate the number of steps in the absolute refractory period
                allNeuronsDevice->numStepsInRefractoryPeriod_[idx] = static_cast<int> ( allNeuronsDevice->Trefract_[idx] / deltaT + 0.5 );

                // reset to 'Vreset'
                vm = allNeuronsDevice->Vreset_[idx];

                // notify outgoing synapses of spike
                BGSIZE synapseCounts = synapseIndexMapDevice->outgoingSynapseCount_[idx];
                if (synapseCounts != 0) {
                    // get the index of where this neuron's list of synapses are
                    BGSIZE beginIndex = synapseIndexMapDevice->outgoingSynapseBegin_[idx];
                    // get the memory location of where that list begins
                    BGSIZE* outgoingMapBegin = &(synapseIndexMapDevice->outgoingSynapseIndexMap_[beginIndex]);

                    // for each synapse, let them know we have fired
                    for (BGSIZE i = 0; i < synapseCounts; i++) {
                        preSpikingSynapsesSpikeHitDevice(outgoingMapBegin[i], allSynapsesDevice);
                    }
                }

                // notify incomming synapses of spike
                synapseCounts = synapseIndexMapDevice->incomingSynapseCount_[idx];
                if (fAllowBackPropagation && synapseCounts != 0) {
                    // get the index of where this neuron's list of synapses are
                    BGSIZE beginIndex = synapseIndexMapDevice->incomingSynapseBegin_[idx];
                    // get the memory location of where that list begins
                    BGSIZE* incomingMapBegin = &(synapseIndexMapDevice->incomingSynapseIndexMap_[beginIndex]);

                    // for each synapse, let them know we have fired
                    switch (classSynapses_d) {
                    case classAllSTDPSynapses:
                    case classAllDynamicSTDPSynapses:
                        for (BGSIZE i = 0; i < synapseCounts; i++) {
                            postSTDPSynapseSpikeHitDevice(incomingMapBegin[i], static_cast<AllSTDPSynapsesDeviceProperties *>(allSynapsesDevice));
                        } // end for
                        break;

                    case classAllSpikingSynapses:
                    case classAllDSSynapses:
                        for (BGSIZE i = 0; i < synapseCounts; i++) {
                            postSpikingSynapsesSpikeHitDevice(incomingMapBegin[i], allSynapsesDevice);
                        } // end for
                        break;

                    default:
                        assert(false);
                    } // end switch
                }
        } else {
                r_sp += allNeuronsDevice->I0_[idx]; // add IO

                // Random number alg. goes here
                r_sp += (randNoise[idx] * allNeuronsDevice->Inoise_[idx]); // add cheap noise
                vm = allNeuronsDevice->C1_[idx] * r_vm + allNeuronsDevice->C2_[idx] * ( r_sp ); // decay Vm and add inputs
        }

        // clear synaptic input for next time step
        sp = 0;
}

///  CUDA code for advancing izhikevich neurons
///
///  @param[in] totalVertices          Number of neurons.
///  @param[in] maxSynapses           Maximum number of synapses per neuron.
///  @param[in] maxSpikes             Maximum number of spikes per neuron per epoch.
///  @param[in] deltaT                Inner simulation step duration.
///  @param[in] simulationStep        The current simulation step.
///  @param[in] randNoise             Pointer to device random noise array.
///  @param[in] allNeuronsDevice      Pointer to Neuron structures in device memory.
///  @param[in] allSynapsesDevice     Pointer to Synapse structures in device memory.
///  @param[in] edgeIndexMap       Inverse map, which is a table indexed by an input neuron and maps to the synapses that provide input to that neuron.
///  @param[in] fAllowBackPropagation True if back propagaion is allowed.
__global__ void advanceIZHNeuronsDevice( int totalVertices, int maxSynapses, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, float* randNoise, AllIZHNeuronsDeviceProperties* allNeuronsDevice, AllSpikingSynapsesDeviceProperties* allSynapsesDevice, EdgeIndexMap* synapseIndexMapDevice, bool fAllowBackPropagation ) {
        // determine which neuron this thread is processing
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx >= totalVertices )
                return;

        allNeuronsDevice->hasFired_[idx] = false;
        BGFLOAT& sp = allNeuronsDevice->summationMap_[idx];
        BGFLOAT& vm = allNeuronsDevice->Vm_[idx];
        BGFLOAT& a = allNeuronsDevice->Aconst_[idx];
        BGFLOAT& b = allNeuronsDevice->Bconst_[idx];
        BGFLOAT& u = allNeuronsDevice->u_[idx];
        BGFLOAT r_sp = sp;
        BGFLOAT r_vm = vm;
        BGFLOAT r_a = a;
        BGFLOAT r_b = b;
        BGFLOAT r_u = u;

        if ( allNeuronsDevice->numStepsInRefractoryPeriod_[idx] > 0 ) { // is neuron refractory?
                --allNeuronsDevice->numStepsInRefractoryPeriod_[idx];
        } else if ( r_vm >= allNeuronsDevice->Vthresh_[idx] ) { // should it fire?
                int& spikeCount = allNeuronsDevice->spikeCount_[idx];
                int& spikeCountOffset = allNeuronsDevice->spikeCountOffset_[idx];

                // Note that the neuron has fired!
                allNeuronsDevice->hasFired_[idx] = true;

                // record spike time
                int idxSp = (spikeCount + spikeCountOffset) % maxSpikes;
                allNeuronsDevice->spikeHistory_[idx][idxSp] = simulationStep;
                spikeCount++;

                // calculate the number of steps in the absolute refractory period
                allNeuronsDevice->numStepsInRefractoryPeriod_[idx] = static_cast<int> ( allNeuronsDevice->Trefract_[idx] / deltaT + 0.5 );

                // reset to 'Vreset'
                vm = allNeuronsDevice->Cconst_[idx] * 0.001;
                u = r_u + allNeuronsDevice->Dconst_[idx];

                // notify outgoing synapses of spike
                BGSIZE synapseCounts = synapseIndexMapDevice->outgoingSynapseCount_[idx];
                if (synapseCounts != 0) {
                    // get the index of where this neuron's list of synapses are
                    BGSIZE beginIndex = synapseIndexMapDevice->outgoingSynapseBegin_[idx]; 
                    // get the memory location of where that list begins
                    BGSIZE* outgoingMapBegin = &(synapseIndexMapDevice->outgoingSynapseIndexMap_[beginIndex]);
                   
                    // for each synapse, let them know we have fired
                    for (BGSIZE i = 0; i < synapseCounts; i++) {
                        preSpikingSynapsesSpikeHitDevice(outgoingMapBegin[i], allSynapsesDevice);
                    }
                }

                // notify incomming synapses of spike
                synapseCounts = synapseIndexMapDevice->incomingSynapseCount_[idx];
                if (fAllowBackPropagation && synapseCounts != 0) {
                    // get the index of where this neuron's list of synapses are
                    BGSIZE beginIndex = synapseIndexMapDevice->incomingSynapseBegin_[idx];
                    // get the memory location of where that list begins
                    BGSIZE* incomingMapBegin = &(synapseIndexMapDevice->incomingSynapseIndexMap_[beginIndex]);

                    // for each synapse, let them know we have fired
                    switch (classSynapses_d) {
                    case classAllSTDPSynapses:
                    case classAllDynamicSTDPSynapses:
                        for (BGSIZE i = 0; i < synapseCounts; i++) {
                            postSTDPSynapseSpikeHitDevice(incomingMapBegin[i], static_cast<AllSTDPSynapsesDeviceProperties *>(allSynapsesDevice));
                        } // end for
                        break;
                    
                    case classAllSpikingSynapses:
                    case classAllDSSynapses:
                        for (BGSIZE i = 0; i < synapseCounts; i++) {
                            postSpikingSynapsesSpikeHitDevice(incomingMapBegin[i], allSynapsesDevice);
                        } // end for
                        break;

                    default:
                        assert(false);
                    } // end switch
                }
        } else {
                r_sp += allNeuronsDevice->I0_[idx]; // add IO

                // Random number alg. goes here
                r_sp += (randNoise[idx] * allNeuronsDevice->Inoise_[idx]); // add cheap noise

                BGFLOAT Vint = r_vm * 1000;

                // Izhikevich model integration step
                BGFLOAT Vb = Vint + allNeuronsDevice->C3_[idx] * (0.04 * Vint * Vint + 5 * Vint + 140 - u);
                u = r_u + allNeuronsDevice->C3_[idx] * r_a * (r_b * Vint - r_u);

                vm = Vb * 0.001 + allNeuronsDevice->C2_[idx] * r_sp;  // add inputs
        }

        // clear synaptic input for next time step
        sp = 0;
}
///@}
