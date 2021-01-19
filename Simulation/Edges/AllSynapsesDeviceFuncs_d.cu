/**
 * @file AllSynapsesDeviceFuncs_d.cu
 * 
 * @ingroup Simulation/Edges
 *
 * @brief
 */

#include <vector>

#include "AllSynapsesDeviceFuncs.h"
#include "AllSynapses.h"
#include "AllSTDPSynapses.h"
#include "AllDynamicSTDPSynapses.h"

/******************************************
 * @name Device Functions for utility
******************************************/
///@{

/// Return 1 if originating neuron is excitatory, -1 otherwise.
///
/// @param[in] t  synapseType I to I, I to E, E to I, or E to E
/// @return 1 or -1
__device__ int synSign( synapseType t )
{
        switch ( t )
        {
        case II:
        case IE:
                return -1;
        case EI:
        case EE:
                return 1;
        }

        return 0;
}
///@}

/******************************************
 * @name Device Functions for advanceSynapses
******************************************/
///@{

///  Update PSR (post synapse response)
///
///  @param  allSynapsesDevice  GPU address of the AllSpikingSynapsesDeviceProperties struct
///                             on device memory.
///  @param  iSyn               Index of the synapse to set.
///  @param  simulationStep     The current simulation step.
///  @param  deltaT             Inner simulation step duration.
__device__ void changeSpikingSynapsesPSRDevice(AllSpikingSynapsesDeviceProperties* allSynapsesDevice, const BGSIZE iSyn, const uint64_t simulationStep, const BGFLOAT deltaT)
{
    BGFLOAT &psr = allSynapsesDevice->psr_[iSyn];
    BGFLOAT &W = allSynapsesDevice->W_[iSyn];
    BGFLOAT &decay = allSynapsesDevice->decay_[iSyn];

    psr += ( W / decay );    // calculate psr
}

///  Update PSR (post synapse response)
///
///  @param  allSynapsesDevice  GPU address of the AllDSSynapsesDeviceProperties struct
///                             on device memory.
///  @param  iSyn               Index of the synapse to set.
///  @param  simulationStep     The current simulation step.
///  @param  deltaT             Inner simulation step duration.
__device__ void changeDSSynapsePSRDevice(AllDSSynapsesDeviceProperties* allSynapsesDevice, const BGSIZE iSyn, const uint64_t simulationStep, const BGFLOAT deltaT)
{
    //assert( iSyn < allSynapsesDevice->maxSynapsesPerNeuron * allSynapsesDevice->countNeurons_ );

    uint64_t &lastSpike = allSynapsesDevice->lastSpike_[iSyn];
    BGFLOAT &r = allSynapsesDevice->r_[iSyn];
    BGFLOAT &u = allSynapsesDevice->u_[iSyn];
    BGFLOAT D = allSynapsesDevice->D_[iSyn];
    BGFLOAT F = allSynapsesDevice->F_[iSyn];
    BGFLOAT U = allSynapsesDevice->U_[iSyn];
    BGFLOAT W = allSynapsesDevice->W_[iSyn];
    BGFLOAT &psr = allSynapsesDevice->psr_[iSyn];
    BGFLOAT decay = allSynapsesDevice->decay_[iSyn];

    // adjust synapse parameters
    if (lastSpike != ULONG_MAX) {
            BGFLOAT isi = (simulationStep - lastSpike) * deltaT ;
            r = 1 + ( r * ( 1 - u ) - 1 ) * exp( -isi / D );
            u = U + u * ( 1 - U ) * exp( -isi / F );
    }
    psr += ( ( W / decay ) * u * r );// calculate psr
    lastSpike = simulationStep; // record the time of the spike
}
    
///  Checks if there is an input spike in the queue.
///
///  @param[in] allSynapsesDevice     GPU address of AllSpikingSynapsesDeviceProperties structures 
///                                   on device memory.
///  @param[in] iSyn                  Index of the Synapse to check.
///
///  @return true if there is an input spike event.
__device__ bool isSpikingSynapsesSpikeQueueDevice(AllSpikingSynapsesDeviceProperties* allSynapsesDevice, BGSIZE iSyn)
{
    uint32_t &delayQueue = allSynapsesDevice->delayQueue_[iSyn];
    int &delayIdx = allSynapsesDevice->delayIndex_[iSyn];
    int delayQueueLength = allSynapsesDevice->delayQueueLength_[iSyn];

    uint32_t delayMask = (0x1 << delayIdx);
    bool isFired = delayQueue & (delayMask);
    delayQueue &= ~(delayMask);
    if ( ++delayIdx >= delayQueueLength ) {
            delayIdx = 0;
    }

    return isFired;
}
   
///  Adjust synapse weight according to the Spike-timing-dependent synaptic modification
///  induced by natural spike trains
///
///  @param  allSynapsesDevice    GPU address of the AllSTDPSynapsesDeviceProperties structures 
///                               on device memory.
///  @param  iSyn                 Index of the synapse to set.
///  @param  delta                Pre/post synaptic spike interval.
///  @param  epost                Params for the rule given in Froemke and Dan (2002).
///  @param  epre                 Params for the rule given in Froemke and Dan (2002).
__device__ void stdpLearningDevice(AllSTDPSynapsesDeviceProperties* allSynapsesDevice, const BGSIZE iSyn, double delta, double epost, double epre)
{
    BGFLOAT STDPgap = allSynapsesDevice->STDPgap_[iSyn];
    BGFLOAT muneg = allSynapsesDevice->muneg_[iSyn];
    BGFLOAT mupos = allSynapsesDevice->mupos_[iSyn];
    BGFLOAT tauneg = allSynapsesDevice->tauneg_[iSyn];
    BGFLOAT taupos = allSynapsesDevice->taupos_[iSyn];
    BGFLOAT Aneg = allSynapsesDevice->Aneg_[iSyn];
    BGFLOAT Apos = allSynapsesDevice->Apos_[iSyn];
    BGFLOAT Wex = allSynapsesDevice->Wex_[iSyn];
    BGFLOAT &W = allSynapsesDevice->W_[iSyn];
    synapseType type = allSynapsesDevice->type_[iSyn];
    BGFLOAT dw;

    if (delta < -STDPgap) {
        // Depression
        dw = pow(fabs(W) / Wex,  muneg) * Aneg * exp(delta / tauneg);  // normalize
    } else if (delta > STDPgap) {
        // Potentiation
        dw = pow(fabs(Wex - fabs(W)) / Wex,  mupos) * Apos * exp(-delta / taupos); // normalize
    } else {
        return;
    }

    // dw is the percentage change in synaptic strength; add 1.0 to become the scaling ratio
    dw = 1.0 + dw * epre * epost;

    // if scaling ratio is less than zero, set it to zero so this synapse, its strength is always zero
    if (dw < 0) {
        dw = 0;
    }

    // current weight multiplies dw (scaling ratio) to generate new weight
    W *= dw;

    // if new weight is bigger than Wex_ (maximum allowed weight), then set it to Wex_
    if (fabs(W) > Wex) {
        W = synSign(type) * Wex;
    }

    // DEBUG_SYNAPSE(
    //     printf("AllSTDPSynapses::stdpLearning:\n");
    //     printf("          iSyn: %d\n", iSyn);
    //     printf("          delta: %f\n", delta);
    //     printf("          epre: %f\n", epre);
    //     printf("          epost: %f\n", epost);
    //     printf("          dw: %f\n", dw);
    //     printf("          W: %f\n\n", W);
    // );
}

///  Checks if there is an input spike in the queue.
///
///  @param[in] allSynapsesDevice     GPU adress of AllSTDPSynapsesDeviceProperties structures 
///                                   on device memory.
///  @param[in] iSyn                  Index of the Synapse to check.
///
///  @return true if there is an input spike event.
__device__ bool isSTDPSynapseSpikeQueuePostDevice(AllSTDPSynapsesDeviceProperties* allSynapsesDevice, BGSIZE iSyn)
{
    uint32_t &delayQueue = allSynapsesDevice->delayQueuePost_[iSyn];
    int &delayIndex = allSynapsesDevice->delayIndexPost_[iSyn];
    int delayQueueLength = allSynapsesDevice->delayQueuePostLength_[iSyn];

    uint32_t delayMask = (0x1 << delayIndex);
    bool isFired = delayQueue & (delayMask);
    delayQueue &= ~(delayMask);
    if ( ++delayIndex >= delayQueueLength ) {
            delayIndex = 0;
    }

    return isFired;
}

///  Gets the spike history of the neuron.
///
///  @param  allNeuronsDevice       GPU address of the allNeurons struct on device memory. 
///  @param  index                  Index of the neuron to get spike history.
///  @param  offIndex               Offset of the history beffer to get.
///                                 -1 will return the last spike.
///  @param  maxSpikes              Maximum number of spikes per neuron per epoch.
///
///  @return Spike history.
__device__ uint64_t getSTDPSynapseSpikeHistoryDevice(AllSpikingNeuronsDeviceProperties* allNeuronsDevice, int index, int offIndex, int maxSpikes)
{
    // offIndex is a minus offset
    int idxSp = (allNeuronsDevice->spikeCount_[index] + allNeuronsDevice->spikeCountOffset_[index] +  maxSpikes + offIndex) % maxSpikes;
    return allNeuronsDevice->spikeHistory_[index][idxSp];
}
///@}

/******************************************
 * @name Global Functions for advanceSynapses
******************************************/
///@{

///  CUDA code for advancing spiking synapses.
///  Perform updating synapses for one time step.
///
///  @param[in] totalSynapseCount  Number of synapses.
///  @param  synapseIndexMapDevice    GPU address of the SynapseIndexMap on device memory.
///  @param[in] simulationStep        The current simulation step.
///  @param[in] deltaT                Inner simulation step duration.
///  @param[in] allSynapsesDevice     Pointer to AllSpikingSynapsesDeviceProperties structures 
///                                   on device memory.
__global__ void advanceSpikingSynapsesDevice ( int totalSynapseCount, SynapseIndexMap* synapseIndexMapDevice, uint64_t simulationStep, const BGFLOAT deltaT, AllSpikingSynapsesDeviceProperties* allSynapsesDevice ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx >= totalSynapseCount )
                return;

                
        BGSIZE iSyn = synapseIndexMapDevice->incomingSynapseIndexMap_[idx];

        BGFLOAT &psr = allSynapsesDevice->psr_[iSyn];
        BGFLOAT decay = allSynapsesDevice->decay_[iSyn];

        // Checks if there is an input spike in the queue.
        bool isFired = isSpikingSynapsesSpikeQueueDevice(allSynapsesDevice, iSyn);

        // is an input in the queue?
        if (isFired) {
                switch (classSynapses_d) {
                case classAllSpikingSynapses:
                       changeSpikingSynapsesPSRDevice(static_cast<AllSpikingSynapsesDeviceProperties*>(allSynapsesDevice), iSyn, simulationStep, deltaT);
                        break;
                case classAllDSSynapses:
                        changeDSSynapsePSRDevice(static_cast<AllDSSynapsesDeviceProperties*>(allSynapsesDevice), iSyn, simulationStep, deltaT);
                        break;
                default:
                        assert(false);
                }
        }
        // decay the post spike response
        psr *= decay;
}

///  CUDA code for advancing STDP synapses.
///  Perform updating synapses for one time step.
///
///  @param[in] totalSynapseCount        Number of synapses.
///  @param[in] synapseIndexMapDevice    GPU address of the SynapseIndexMap on device memory.
///  @param[in] simulationStep           The current simulation step.
///  @param[in] deltaT                   Inner simulation step duration.
///  @param[in] allSynapsesDevice        GPU address of AllSTDPSynapsesDeviceProperties structures 
///                                      on device memory.
///  @param[in] allNeuronsDevice         GPU address of AllNeurons structures on device memory.
///  @param[in] maxSpikes                Maximum number of spikes per neuron per epoch.               
__global__ void advanceSTDPSynapsesDevice ( int totalSynapseCount, SynapseIndexMap* synapseIndexMapDevice, uint64_t simulationStep, const BGFLOAT deltaT, AllSTDPSynapsesDeviceProperties* allSynapsesDevice, AllSpikingNeuronsDeviceProperties* allNeuronsDevice, int maxSpikes ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= totalSynapseCount )
            return;

    BGSIZE iSyn = synapseIndexMapDevice->incomingSynapseIndexMap_[idx];

    // If the synapse is inhibitory or its weight is zero, update synapse state using AllSpikingSynapses::advanceSynapse method
    BGFLOAT &W = allSynapsesDevice->W_[iSyn];
    if(W <= 0.0) {
        BGFLOAT &psr = allSynapsesDevice->psr_[iSyn];
        BGFLOAT decay = allSynapsesDevice->decay_[iSyn];

        // Checks if there is an input spike in the queue.
        bool isFired = isSpikingSynapsesSpikeQueueDevice(allSynapsesDevice, iSyn);

        // is an input in the queue?
        if (isFired) {
                switch (classSynapses_d) {
                case classAllSTDPSynapses:
                        changeSpikingSynapsesPSRDevice(static_cast<AllSpikingSynapsesDeviceProperties*>(allSynapsesDevice), iSyn, simulationStep, deltaT);
                        break;
                case classAllDynamicSTDPSynapses:
                	// Note: we cast void * over the allSynapsesDevice, then recast it, 
                	// because AllDSSynapsesDeviceProperties inherited properties from 
                	// the AllDSSynapsesDeviceProperties and the AllSTDPSynapsesDeviceProperties.
			changeDSSynapsePSRDevice(static_cast<AllDSSynapsesDeviceProperties*>((void *)allSynapsesDevice), iSyn, simulationStep, deltaT);
                        break;
                default:
                        assert(false);
                }
        }
        // decay the post spike response
        psr *= decay;
        return;
    }

    BGFLOAT &decay = allSynapsesDevice->decay_[iSyn];
    BGFLOAT &psr = allSynapsesDevice->psr_[iSyn];

    // is an input in the queue?
    bool fPre = isSpikingSynapsesSpikeQueueDevice(allSynapsesDevice, iSyn);
    bool fPost = isSTDPSynapseSpikeQueuePostDevice(allSynapsesDevice, iSyn);
    if (fPre || fPost) {
        BGFLOAT &tauspre = allSynapsesDevice->tauspre_[iSyn];
        BGFLOAT &tauspost = allSynapsesDevice->tauspost_[iSyn];
        BGFLOAT &taupos = allSynapsesDevice->taupos_[iSyn];
        BGFLOAT &tauneg = allSynapsesDevice->tauneg_[iSyn];
        int &totalDelay = allSynapsesDevice->totalDelay_[iSyn];
        bool &useFroemkeDanSTDP = allSynapsesDevice->useFroemkeDanSTDP_[iSyn];

        // pre and post neurons index
        int idxPre = allSynapsesDevice->sourceNeuronIndex_[iSyn];
        int idxPost = allSynapsesDevice->destNeuronIndex_[iSyn];
        int64_t spikeHistory, spikeHistory2;
        BGFLOAT delta;
        BGFLOAT epre, epost;

        if (fPre) {     // preSpikeHit
            // spikeCount points to the next available position of spike_history,
            // so the getSpikeHistory w/offset = -2 will return the spike time 
            // just one before the last spike.
            spikeHistory = getSTDPSynapseSpikeHistoryDevice(allNeuronsDevice, idxPre, -2, maxSpikes);
            if (spikeHistory > 0 && useFroemkeDanSTDP) {
                // delta will include the transmission delay
                delta = static_cast<BGFLOAT>(simulationStep - spikeHistory) * deltaT;
                epre = 1.0 - exp(-delta / tauspre);
            } else {
                epre = 1.0;
            }

            // call the learning function stdpLearning() for each pair of
            // pre-post spikes
            int offIndex = -1;  // last spike
            while (true) {
                spikeHistory = getSTDPSynapseSpikeHistoryDevice(allNeuronsDevice, idxPost, offIndex, maxSpikes);
                if (spikeHistory == ULONG_MAX)
                    break;
                // delta is the spike interval between pre-post spikes
                delta = -static_cast<BGFLOAT>(simulationStep - spikeHistory) * deltaT;

                if (delta <= -3.0 * tauneg)
                    break;
                if (useFroemkeDanSTDP) {
                    spikeHistory2 = getSTDPSynapseSpikeHistoryDevice(allNeuronsDevice, idxPost, offIndex-1, maxSpikes);
                    if (spikeHistory2 == ULONG_MAX)
                        break;
                    epost = 1.0 - exp(-(static_cast<BGFLOAT>(spikeHistory - spikeHistory2) * deltaT) / tauspost);
                } else {
                    epost = 1.0;
                }
                stdpLearningDevice(allSynapsesDevice, iSyn, delta, epost, epre);
                --offIndex;
            }

            switch (classSynapses_d) {
            case classAllSTDPSynapses:
                changeSpikingSynapsesPSRDevice(static_cast<AllSpikingSynapsesDeviceProperties*>(allSynapsesDevice), iSyn, simulationStep, deltaT);
                break;
            case classAllDynamicSTDPSynapses:
                // Note: we cast void * over the allSynapsesDevice, then recast it, 
                // because AllDSSynapsesDeviceProperties inherited properties from 
                // the AllDSSynapsesDeviceProperties and the AllSTDPSynapsesDeviceProperties.
                changeDSSynapsePSRDevice(static_cast<AllDSSynapsesDeviceProperties*>((void *)allSynapsesDevice), iSyn, simulationStep, deltaT);
                break;
            default:
                assert(false);
            }
        }

        if (fPost) {    // postSpikeHit
            // spikeCount points to the next available position of spike_history,
            // so the getSpikeHistory w/offset = -2 will return the spike time
            // just one before the last spike.
            spikeHistory = getSTDPSynapseSpikeHistoryDevice(allNeuronsDevice, idxPost, -2, maxSpikes);
            if (spikeHistory > 0 && useFroemkeDanSTDP) {
                // delta will include the transmission delay
                delta = static_cast<BGFLOAT>(simulationStep - spikeHistory) * deltaT;
                epost = 1.0 - exp(-delta / tauspost);
            } else {
                epost = 1.0;
            }

            // call the learning function stdpLearning() for each pair of
            // post-pre spikes
            int offIndex = -1;  // last spike
            while (true) {
                spikeHistory = getSTDPSynapseSpikeHistoryDevice(allNeuronsDevice, idxPre, offIndex, maxSpikes);
                if (spikeHistory == ULONG_MAX)
                    break;

                if(spikeHistory + totalDelay > simulationStep) {
                    --offIndex;
                    continue;
                }
                // delta is the spike interval between post-pre spikes
                delta = static_cast<BGFLOAT>(simulationStep - spikeHistory - totalDelay) * deltaT;
                

                if (delta >= 3.0 * taupos)
                    break;
                if (useFroemkeDanSTDP) {
                    spikeHistory2 = getSTDPSynapseSpikeHistoryDevice(allNeuronsDevice, idxPre, offIndex-1, maxSpikes);
                    if (spikeHistory2 == ULONG_MAX)
                        break;
                    epre = 1.0 - exp(-(static_cast<BGFLOAT>(spikeHistory - spikeHistory2) * deltaT) / tauspre);
                } else {
                    epre = 1.0;
                }
                stdpLearningDevice(allSynapsesDevice, iSyn, delta, epost, epre);
                --offIndex;
            }
        }
    }

    // decay the post spike response
    psr *= decay;
}
///@}

/******************************************
 * @name Device Functions for createSynapse
******************************************/
///@{

///  Create a Spiking Synapse and connect it to the model.
///
///  @param allSynapsesDevice    GPU address of the AllSpikingSynapsesDeviceProperties structures 
///                              on device memory.
///  @param neuronIndex          Index of the source neuron.
///  @param synapseOffset        Offset (into neuronIndex's) of the Synapse to create.
///  @param srcNeuron            Coordinates of the source Neuron.
///  @param destNeuron           Coordinates of the destination Neuron.
///  @param sumPoint             Pointer to the summation point.
///  @param deltaT               The time step size.
///  @param type                 Type of the Synapse to create.
__device__ void createSpikingSynapse(AllSpikingSynapsesDeviceProperties* allSynapsesDevice, const int neuronIndex, const int synapseOffset, int sourceIndex, int destIndex, BGFLOAT *sumPoint, const BGFLOAT deltaT, synapseType type)
{
    BGFLOAT delay;
    BGSIZE maxSynapses = allSynapsesDevice->maxSynapsesPerNeuron_;
    BGSIZE iSyn = maxSynapses * neuronIndex + synapseOffset;

    allSynapsesDevice->inUse_[iSyn] = true;
    allSynapsesDevice->destNeuronIndex_[iSyn] = destIndex;
    allSynapsesDevice->sourceNeuronIndex_[iSyn] = sourceIndex;
    allSynapsesDevice->W_[iSyn] = synSign(type) * 10.0e-9;
    
    allSynapsesDevice->delayQueue_[iSyn] = 0;
    allSynapsesDevice->delayIndex_[iSyn] = 0;
    allSynapsesDevice->delayQueueLength_[iSyn] = LENGTH_OF_DELAYQUEUE;

    allSynapsesDevice->psr_[iSyn] = 0.0;
    allSynapsesDevice->type_[iSyn] = type;

    allSynapsesDevice->tau_[iSyn] = DEFAULT_tau;

    BGFLOAT tau;
    switch (type) {
        case II:
            tau = 6e-3;
            delay = 0.8e-3;
            break;
        case IE:
            tau = 6e-3;
            delay = 0.8e-3;
            break;
        case EI:
            tau = 3e-3;
            delay = 0.8e-3;
            break;
        case EE:
            tau = 3e-3;
            delay = 1.5e-3;
            break;
        default:
            break;
    }

    allSynapsesDevice->tau_[iSyn] = tau;
    allSynapsesDevice->decay_[iSyn] = exp( -deltaT / tau );
    allSynapsesDevice->totalDelay_[iSyn] = static_cast<int>( delay / deltaT ) + 1;

    uint32_t size = allSynapsesDevice->totalDelay_[iSyn] / ( sizeof(uint8_t) * 8 ) + 1;
    assert( size <= BYTES_OF_DELAYQUEUE );
}

///  Create a DS Synapse and connect it to the model.
///
///  @param allSynapsesDevice    GPU address of the AllDSSynapsesDeviceProperties structures 
///                              on device memory.
///  @param neuronIndex          Index of the source neuron.
///  @param synapseOffset        Offset (into neuronIndex's) of the Synapse to create.
///  @param srcNeuron            Coordinates of the source Neuron.
///  @param destNeuron           Coordinates of the destination Neuron.
///  @param sumPoint             Pointer to the summation point.
///  @param deltaT               The time step size.
///  @param type                 Type of the Synapse to create.
__device__ void createDSSynapse(AllDSSynapsesDeviceProperties* allSynapsesDevice, const int neuronIndex, const int synapseOffset, int sourceIndex, int destIndex, BGFLOAT *sumPoint, const BGFLOAT deltaT, synapseType type)
{
    BGFLOAT delay;
    BGSIZE maxSynapses = allSynapsesDevice->maxSynapsesPerNeuron_;
    BGSIZE iSyn = maxSynapses * neuronIndex + synapseOffset;

    allSynapsesDevice->inUse_[iSyn] = true;
    allSynapsesDevice->destNeuronIndex_[iSyn] = destIndex;
    allSynapsesDevice->sourceNeuronIndex_[iSyn] = sourceIndex;
    allSynapsesDevice->W_[iSyn] = synSign(type) * 10.0e-9;

    allSynapsesDevice->delayQueue_[iSyn] = 0;
    allSynapsesDevice->delayIndex_[iSyn] = 0;
    allSynapsesDevice->delayQueueLength_[iSyn] = LENGTH_OF_DELAYQUEUE;

    allSynapsesDevice->psr_[iSyn] = 0.0;
    allSynapsesDevice->r_[iSyn] = 1.0;
    allSynapsesDevice->u_[iSyn] = 0.4;     // DEFAULT_U
    allSynapsesDevice->lastSpike_[iSyn] = ULONG_MAX;
    allSynapsesDevice->type_[iSyn] = type;

    allSynapsesDevice->U_[iSyn] = DEFAULT_U;
    allSynapsesDevice->tau_[iSyn] = DEFAULT_tau;

    BGFLOAT U;
    BGFLOAT D;
    BGFLOAT F;
    BGFLOAT tau;
    switch (type) {
        case II:
            U = 0.32;
            D = 0.144;
            F = 0.06;
            tau = 6e-3;
            delay = 0.8e-3;
            break;
        case IE:
            U = 0.25;
            D = 0.7;
            F = 0.02;
            tau = 6e-3;
            delay = 0.8e-3;
            break;
        case EI:
            U = 0.05;
            D = 0.125;
            F = 1.2;
            tau = 3e-3;
            delay = 0.8e-3;
            break;
        case EE:
            U = 0.5;
            D = 1.1;
            F = 0.05;
            tau = 3e-3;
            delay = 1.5e-3;
            break;
        default:
            break;
    }

    allSynapsesDevice->U_[iSyn] = U;
    allSynapsesDevice->D_[iSyn] = D;
    allSynapsesDevice->F_[iSyn] = F;

    allSynapsesDevice->tau_[iSyn] = tau;
    allSynapsesDevice->decay_[iSyn] = exp( -deltaT / tau );
    allSynapsesDevice->totalDelay_[iSyn] = static_cast<int>( delay / deltaT ) + 1;

    uint32_t size = allSynapsesDevice->totalDelay_[iSyn] / ( sizeof(uint8_t) * 8 ) + 1;
    assert( size <= BYTES_OF_DELAYQUEUE );
}

///  Create a Synapse and connect it to the model.
///
///  @param allSynapsesDevice    GPU address of the AllSTDPSynapsesDeviceProperties structures 
///                              on device memory.
///  @param neuronIndex          Index of the source neuron.
///  @param synapseOffset        Offset (into neuronIndex's) of the Synapse to create.
///  @param srcNeuron            Coordinates of the source Neuron.
///  @param destNeuron           Coordinates of the destination Neuron.
///  @param sumPoint             Pointer to the summation point.
///  @param deltaT               The time step size.
///  @param type                 Type of the Synapse to create.
__device__ void createSTDPSynapse(AllSTDPSynapsesDeviceProperties* allSynapsesDevice, const int neuronIndex, const int synapseOffset, int sourceIndex, int destIndex, BGFLOAT *sumPoint, const BGFLOAT deltaT, synapseType type)
{
    BGFLOAT delay;
    BGSIZE maxSynapses = allSynapsesDevice->maxSynapsesPerNeuron_;
    BGSIZE iSyn = maxSynapses * neuronIndex + synapseOffset;

    allSynapsesDevice->inUse_[iSyn] = true;
    allSynapsesDevice->destNeuronIndex_[iSyn] = destIndex;
    allSynapsesDevice->sourceNeuronIndex_[iSyn] = sourceIndex;
    allSynapsesDevice->W_[iSyn] = synSign(type) * 10.0e-9;

    allSynapsesDevice->delayQueue_[iSyn] = 0;
    allSynapsesDevice->delayIndex_[iSyn] = 0;
    allSynapsesDevice->delayQueueLength_[iSyn] = LENGTH_OF_DELAYQUEUE;

    allSynapsesDevice->psr_[iSyn] = 0.0;
    allSynapsesDevice->type_[iSyn] = type;

    allSynapsesDevice->tau_[iSyn] = DEFAULT_tau;

    BGFLOAT tau;
    switch (type) {
        case II:
            tau = 6e-3;
            delay = 0.8e-3;
            break;
        case IE:
            tau = 6e-3;
            delay = 0.8e-3;
            break;
        case EI:
            tau = 3e-3;
            delay = 0.8e-3;
            break;
        case EE:
            tau = 3e-3;
            delay = 1.5e-3;
            break;
        default:
            break;
    }

    allSynapsesDevice->tau_[iSyn] = tau;
    allSynapsesDevice->decay_[iSyn] = exp( -deltaT / tau );
    allSynapsesDevice->totalDelay_[iSyn] = static_cast<int>( delay / deltaT ) + 1;

    uint32_t size = allSynapsesDevice->totalDelay_[iSyn] / ( sizeof(uint8_t) * 8 ) + 1;
    assert( size <= BYTES_OF_DELAYQUEUE );

    // May 1st 2020 
    // Use constants from Froemke and Dan (2002). 
    // Spike-timing-dependent synaptic modification induced by natural spike trains. Nature 416 (3/2002)
    allSynapsesDevice->Apos_[iSyn] = 1.01;
    allSynapsesDevice->Aneg_[iSyn] = -0.52;
    allSynapsesDevice->STDPgap_[iSyn] = 2e-3;

    allSynapsesDevice->totalDelayPost_[iSyn] = 0;

    allSynapsesDevice->tauspost_[iSyn] = 75e-3;
    allSynapsesDevice->tauspre_[iSyn] = 34e-3;

    allSynapsesDevice->taupos_[iSyn] = 14.8e-3;
    allSynapsesDevice->tauneg_[iSyn] = 33.8e-3;
    allSynapsesDevice->Wex_[iSyn] = 5.0265e-7;

    allSynapsesDevice->mupos_[iSyn] = 0;
    allSynapsesDevice->muneg_[iSyn] = 0;

    allSynapsesDevice->useFroemkeDanSTDP_[iSyn] = false;
}

///  Create a Synapse and connect it to the model.
///
///  @param allSynapsesDevice    GPU address of the AllDynamicSTDPSynapsesDeviceProperties structures 
///                              on device memory.
///  @param neuronIndex          Index of the source neuron.
///  @param synapseOffset        Offset (into neuronIndex's) of the Synapse to create.
///  @param srcNeuron            Coordinates of the source Neuron.
///  @param destNeuron           Coordinates of the destination Neuron.
///  @param sumPoint             Pointer to the summation point.
///  @param deltaT               The time step size.
///  @param type                 Type of the Synapse to create.
__device__ void createDynamicSTDPSynapse(AllDynamicSTDPSynapsesDeviceProperties* allSynapsesDevice, const int neuronIndex, const int synapseOffset, int sourceIndex, int destIndex, BGFLOAT *sumPoint, const BGFLOAT deltaT, synapseType type)
{
    BGFLOAT delay;
    BGSIZE maxSynapses = allSynapsesDevice->maxSynapsesPerNeuron_;
    BGSIZE iSyn = maxSynapses * neuronIndex + synapseOffset;

    allSynapsesDevice->inUse_[iSyn] = true;
    allSynapsesDevice->destNeuronIndex_[iSyn] = destIndex;
    allSynapsesDevice->sourceNeuronIndex_[iSyn] = sourceIndex;
    allSynapsesDevice->W_[iSyn] = synSign(type) * 10.0e-9;

    allSynapsesDevice->delayQueue_[iSyn] = 0;
    allSynapsesDevice->delayIndex_[iSyn] = 0;
    allSynapsesDevice->delayQueueLength_[iSyn] = LENGTH_OF_DELAYQUEUE;

    allSynapsesDevice->psr_[iSyn] = 0.0;
    allSynapsesDevice->r_[iSyn] = 1.0;
    allSynapsesDevice->u_[iSyn] = 0.4;     // DEFAULT_U
    allSynapsesDevice->lastSpike_[iSyn] = ULONG_MAX;
    allSynapsesDevice->type_[iSyn] = type;

    allSynapsesDevice->U_[iSyn] = DEFAULT_U;
    allSynapsesDevice->tau_[iSyn] = DEFAULT_tau;

    BGFLOAT U;
    BGFLOAT D;
    BGFLOAT F;
    BGFLOAT tau;
    switch (type) {
        case II:
            U = 0.32;
            D = 0.144;
            F = 0.06;
            tau = 6e-3;
            delay = 0.8e-3;
            break;
        case IE:
            U = 0.25;
            D = 0.7;
            F = 0.02;
            tau = 6e-3;
            delay = 0.8e-3;
            break;
        case EI:
            U = 0.05;
            D = 0.125;
            F = 1.2;
            tau = 3e-3;
            delay = 0.8e-3;
            break;
        case EE:
            U = 0.5;
            D = 1.1;
            F = 0.05;
            tau = 3e-3;
            delay = 1.5e-3;
            break;
        default:
            break;
    }

    allSynapsesDevice->U_[iSyn] = U;
    allSynapsesDevice->D_[iSyn] = D;
    allSynapsesDevice->F_[iSyn] = F;

    allSynapsesDevice->tau_[iSyn] = tau;
    allSynapsesDevice->decay_[iSyn] = exp( -deltaT / tau );
    allSynapsesDevice->totalDelay_[iSyn] = static_cast<int>( delay / deltaT ) + 1;

    uint32_t size = allSynapsesDevice->totalDelay_[iSyn] / ( sizeof(uint8_t) * 8 ) + 1;
    assert( size <= BYTES_OF_DELAYQUEUE );

    // May 1st 2020 
    // Use constants from Froemke and Dan (2002). 
    // Spike-timing-dependent synaptic modification induced by natural spike trains. Nature 416 (3/2002)
    allSynapsesDevice->Apos_[iSyn] = 1.01;
    allSynapsesDevice->Aneg_[iSyn] = -0.52;
    allSynapsesDevice->STDPgap_[iSyn] = 2e-3;

    allSynapsesDevice->totalDelayPost_[iSyn] = 0;

    allSynapsesDevice->tauspost_[iSyn] = 75e-3;
    allSynapsesDevice->tauspre_[iSyn] = 34e-3;

    allSynapsesDevice->taupos_[iSyn] = 14.8e-3;
    allSynapsesDevice->tauneg_[iSyn] = 33.8e-3;
    allSynapsesDevice->Wex_[iSyn] = 5.0265e-7;

    allSynapsesDevice->mupos_[iSyn] = 0;
    allSynapsesDevice->muneg_[iSyn] = 0;

    allSynapsesDevice->useFroemkeDanSTDP_[iSyn] = false;
}

/// Adds a synapse to the network.  Requires the locations of the source and
/// destination neurons.
///
/// @param allSynapsesDevice      Pointer to the AllSpikingSynapsesDeviceProperties structures 
///                               on device memory.
/// @param type                   Type of the Synapse to create.
/// @param srcNeuron            Coordinates of the source Neuron.
/// @param destNeuron           Coordinates of the destination Neuron.
///
/// @param sumPoint              Pointer to the summation point.
/// @param deltaT                 The time step size.
/// @param W_d                    Array of synapse weight.
/// @param numNeurons            The number of neurons.
__device__ void addSpikingSynapse(AllSpikingSynapsesDeviceProperties* allSynapsesDevice, synapseType type, const int srcNeuron, const int destNeuron, int sourceIndex, int destIndex, BGFLOAT *sumPoint, const BGFLOAT deltaT, BGFLOAT* W_d, int numNeurons)
{
    if (allSynapsesDevice->synapseCounts_[destNeuron] >= allSynapsesDevice->maxSynapsesPerNeuron_) {
        return; // TODO: ERROR!
    }

    // add it to the list
    BGSIZE synapseIndex;
    BGSIZE maxSynapses = allSynapsesDevice->maxSynapsesPerNeuron_;
    BGSIZE synapseBegin = maxSynapses * destNeuron;
    for (synapseIndex = 0; synapseIndex < maxSynapses; synapseIndex++) {
        if (!allSynapsesDevice->inUse_[synapseBegin + synapseIndex]) {
            break;
        }
    }

    allSynapsesDevice->synapseCounts_[destNeuron]++;

    // create a synapse
    switch (classSynapses_d) {
    case classAllSpikingSynapses:
        createSpikingSynapse(allSynapsesDevice, destNeuron, synapseIndex, sourceIndex, destIndex, sumPoint, deltaT, type );
        break;
    case classAllDSSynapses:
        createDSSynapse(static_cast<AllDSSynapsesDeviceProperties *>(allSynapsesDevice), destNeuron, synapseIndex, sourceIndex, destIndex, sumPoint, deltaT, type );
        break;
    case classAllSTDPSynapses:
        createSTDPSynapse(static_cast<AllSTDPSynapsesDeviceProperties *>(allSynapsesDevice), destNeuron, synapseIndex, sourceIndex, destIndex, sumPoint, deltaT, type );
        break;
    case classAllDynamicSTDPSynapses:
        createDynamicSTDPSynapse(static_cast<AllDynamicSTDPSynapsesDeviceProperties *>(allSynapsesDevice), destNeuron, synapseIndex, sourceIndex, destIndex, sumPoint, deltaT, type );
        break;
    default:
        assert(false);
    }
    allSynapsesDevice->W_[synapseBegin + synapseIndex] = W_d[srcNeuron * numNeurons + destNeuron] * synSign(type) * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;
}

/// Remove a synapse from the network.
///
/// @param[in] allSynapsesDevice      Pointer to the AllSpikingSynapsesDeviceProperties structures 
///                                   on device memory.
/// @param neuronIndex               Index of a neuron.
/// @param synapseOffset             Offset into neuronIndex's synapses.
/// @param[in] maxSynapses            Maximum number of synapses per neuron.
__device__ void eraseSpikingSynapse( AllSpikingSynapsesDeviceProperties* allSynapsesDevice, const int neuronIndex, const int synapseOffset, int maxSynapses )
{
    BGSIZE iSync = maxSynapses * neuronIndex + synapseOffset;
    allSynapsesDevice->synapseCounts_[neuronIndex]--;
    allSynapsesDevice->inUse_[iSync] = false;
    allSynapsesDevice->W_[iSync] = 0;
}

/// Returns the type of synapse at the given coordinates
///
/// @param[in] allNeuronsDevice          Pointer to the Neuron structures in device memory.
/// @param srcNeuron             Index of the source neuron.
/// @param destNeuron            Index of the destination neuron.
__device__ synapseType synType( neuronType* neuronTypeMap_d, const int srcNeuron, const int destNeuron )
{
    if ( neuronTypeMap_d[srcNeuron] == INH && neuronTypeMap_d[destNeuron] == INH )
        return II;
    else if ( neuronTypeMap_d[srcNeuron] == INH && neuronTypeMap_d[destNeuron] == EXC )
        return IE;
    else if ( neuronTypeMap_d[srcNeuron] == EXC && neuronTypeMap_d[destNeuron] == INH )
        return EI;
    else if ( neuronTypeMap_d[srcNeuron] == EXC && neuronTypeMap_d[destNeuron] == EXC )
        return EE;

    return STYPE_UNDEF;

}
///@}

/******************************************
 * @name Global Functions for updateSynapses
******************************************/
///@{

/// Adjust the strength of the synapse or remove it from the synapse map if it has gone below
/// zero.
///
/// @param[in] numNeurons        Number of neurons.
/// @param[in] deltaT             The time step size.
/// @param[in] W_d                Array of synapse weight.
/// @param[in] maxSynapses        Maximum number of synapses per neuron.
/// @param[in] allNeuronsDevice   Pointer to the Neuron structures in device memory.
/// @param[in] allSynapsesDevice  Pointer to the Synapse structures in device memory.
__global__ void updateSynapsesWeightsDevice( int numNeurons, BGFLOAT deltaT, BGFLOAT* W_d, int maxSynapses, AllSpikingNeuronsDeviceProperties* allNeuronsDevice, AllSpikingSynapsesDeviceProperties* allSynapsesDevice, neuronType* neuronTypeMap_d )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= numNeurons )
        return;

    int adjusted = 0;
    //int could_have_been_removed = 0; // TODO: use this value
    int removed = 0;
    int added = 0;

    // Scale and add sign to the areas
    // visit each neuron 'a'
    int destNeuron = idx;

    // and each destination neuron 'b'
    for (int srcNeuron = 0; srcNeuron < numNeurons; srcNeuron++) {
        // visit each synapse at (xa,ya)
        bool connected = false;
        synapseType type = synType(neuronTypeMap_d, srcNeuron, destNeuron);

        // for each existing synapse
        BGSIZE existing_synapses = allSynapsesDevice->synapseCounts_[destNeuron];
        int existingSynapsesChecked = 0;
        for (BGSIZE synapseIndex = 0; (existingSynapsesChecked < existing_synapses) && !connected; synapseIndex++) {
            BGSIZE iSyn = maxSynapses * destNeuron + synapseIndex;
            if (allSynapsesDevice->inUse_[iSyn] == true) {
                // if there is a synapse between a and b
                if (allSynapsesDevice->sourceNeuronIndex_[iSyn] == srcNeuron) {
                    connected = true;
                    adjusted++;

                    // adjust the strength of the synapse or remove
                    // it from the synapse map if it has gone below
                    // zero.
                    if (W_d[srcNeuron * numNeurons + destNeuron] <= 0) {
                        removed++;
                        eraseSpikingSynapse(allSynapsesDevice, destNeuron, synapseIndex, maxSynapses);
                    } else {
                        // adjust
                        // g_synapseStrengthAdjustmentConstant is 1.0e-8;
                        allSynapsesDevice->W_[iSyn] = W_d[srcNeuron * numNeurons
                            + destNeuron] * synSign(type) * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;
                    }
                }
                existingSynapsesChecked++;
            }
        }

        // if not connected and weight(a,b) > 0, add a new synapse from a to b
        if (!connected && (W_d[srcNeuron * numNeurons +  destNeuron] > 0)) {
            // locate summation point
            BGFLOAT* sumPoint = &( allNeuronsDevice->summationMap_[destNeuron] );
            added++;

            addSpikingSynapse(allSynapsesDevice, type, srcNeuron, destNeuron, srcNeuron, destNeuron, sumPoint, deltaT, W_d, numNeurons);
        }
    }
}


/// Adds a synapse to the network.  Requires the locations of the source and
/// destination neurons.
///
/// @param allSynapsesDevice      Pointer to the Synapse structures in device memory.
/// @param pSummationMap          Pointer to the summation point.
/// @param deltaT                 The simulation time step size.
/// @param weight                 Synapse weight.
__global__ void initSynapsesDevice( int n, AllDSSynapsesDeviceProperties* allSynapsesDevice, BGFLOAT *pSummationMap, const BGFLOAT deltaT, BGFLOAT weight )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n )
        return;

    // create a synapse
    int neuronIndex = idx;
    BGFLOAT* sumPoint = &( pSummationMap[neuronIndex] );
    synapseType type = allSynapsesDevice->type_[neuronIndex];
    createDSSynapse(allSynapsesDevice, neuronIndex, 0, 0, neuronIndex, sumPoint, deltaT, type );
    allSynapsesDevice->W_[neuronIndex] = weight * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;
}
///@}

