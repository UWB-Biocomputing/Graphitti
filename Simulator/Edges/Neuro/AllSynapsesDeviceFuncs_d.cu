/**
 * @file AllSynapsesDeviceFuncs_d.cu
 * 
 * @ingroup Simulator/Edges
 *
 * @brief Device functions for synapse data
 */

#include <vector>

#include "AllSynapsesDeviceFuncs.h"
#include "AllNeuroEdges.h"
#include "AllSTDPSynapses.h"
#include "AllDynamicSTDPSynapses.h"

/******************************************
 * @name Device Functions for utility
******************************************/
///@{

/// Return 1 if originating neuron is excitatory, -1 otherwise.
///
/// @param[in] t  edgeType I to I, I to E, E to I, or E to E
/// @return 1 or -1
__device__ int edgSign( edgeType t )
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
 * @name Device Functions for advanceEdges
******************************************/
///@{

///  Update PSR (post synapse response)
///
///  @param  allEdgesDevice  GPU address of the AllSpikingSynapsesDeviceProperties struct
///                             on device memory.
///  @param  iEdg               Index of the synapse to set.
///  @param  simulationStep     The current simulation step.
///  @param  deltaT             Inner simulation step duration.
__device__ void changeSpikingSynapsesPSRDevice(AllSpikingSynapsesDeviceProperties* allEdgesDevice, const BGSIZE iEdg, const uint64_t simulationStep, const BGFLOAT deltaT)
{
    BGFLOAT &psr = allEdgesDevice->psr_[iEdg];
    BGFLOAT &W = allEdgesDevice->W_[iEdg];
    BGFLOAT &decay = allEdgesDevice->decay_[iEdg];

    psr += ( W / decay );    // calculate psr
}

///  Update PSR (post synapse response)
///
///  @param  allEdgesDevice  GPU address of the AllDSSynapsesDeviceProperties struct
///                             on device memory.
///  @param  iEdg               Index of the synapse to set.
///  @param  simulationStep     The current simulation step.
///  @param  deltaT             Inner simulation step duration.
__device__ void changeDSSynapsePSRDevice(AllDSSynapsesDeviceProperties* allEdgesDevice, const BGSIZE iEdg, const uint64_t simulationStep, const BGFLOAT deltaT)
{
    //assert( iEdg < allEdgesDevice->maxEdgesPerVertex * allEdgesDevice->countVertices_ );

    uint64_t &lastSpike = allEdgesDevice->lastSpike_[iEdg];
    BGFLOAT &r = allEdgesDevice->r_[iEdg];
    BGFLOAT &u = allEdgesDevice->u_[iEdg];
    BGFLOAT D = allEdgesDevice->D_[iEdg];
    BGFLOAT F = allEdgesDevice->F_[iEdg];
    BGFLOAT U = allEdgesDevice->U_[iEdg];
    BGFLOAT W = allEdgesDevice->W_[iEdg];
    BGFLOAT &psr = allEdgesDevice->psr_[iEdg];
    BGFLOAT decay = allEdgesDevice->decay_[iEdg];

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
///  @param[in] allEdgesDevice     GPU address of AllSpikingSynapsesDeviceProperties structures 
///                                   on device memory.
///  @param[in] iEdg                  Index of the Synapse to check.
///
///  @return true if there is an input spike event.
__device__ bool isSpikingSynapsesSpikeQueueDevice(AllSpikingSynapsesDeviceProperties* allEdgesDevice, BGSIZE iEdg)
{
    uint32_t &delayQueue = allEdgesDevice->delayQueue_[iEdg];
    int &delayIdx = allEdgesDevice->delayIndex_[iEdg];
    int delayQueueLength = allEdgesDevice->delayQueueLength_[iEdg];

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
///  @param  allEdgesDevice    GPU address of the AllSTDPSynapsesDeviceProperties structures 
///                               on device memory.
///  @param  iEdg                 Index of the synapse to set.
///  @param  delta                Pre/post synaptic spike interval.
///  @param  epost                Params for the rule given in Froemke and Dan (2002).
///  @param  epre                 Params for the rule given in Froemke and Dan (2002).
__device__ void stdpLearningDevice(AllSTDPSynapsesDeviceProperties* allEdgesDevice, const BGSIZE iEdg, double delta, double epost, double epre)
{
    BGFLOAT STDPgap = allEdgesDevice->STDPgap_[iEdg];
    BGFLOAT muneg = allEdgesDevice->muneg_[iEdg];
    BGFLOAT mupos = allEdgesDevice->mupos_[iEdg];
    BGFLOAT tauneg = allEdgesDevice->tauneg_[iEdg];
    BGFLOAT taupos = allEdgesDevice->taupos_[iEdg];
    BGFLOAT Aneg = allEdgesDevice->Aneg_[iEdg];
    BGFLOAT Apos = allEdgesDevice->Apos_[iEdg];
    BGFLOAT Wex = allEdgesDevice->Wex_[iEdg];
    BGFLOAT &W = allEdgesDevice->W_[iEdg];
    edgeType type = allEdgesDevice->type_[iEdg];
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
        W = edgSign(type) * Wex;
    }

    // DEBUG_SYNAPSE(
    //     printf("AllSTDPSynapses::stdpLearning:\n");
    //     printf("          iEdg: %d\n", iEdg);
    //     printf("          delta: %f\n", delta);
    //     printf("          epre: %f\n", epre);
    //     printf("          epost: %f\n", epost);
    //     printf("          dw: %f\n", dw);
    //     printf("          W: %f\n\n", W);
    // );
}

///  Checks if there is an input spike in the queue.
///
///  @param[in] allEdgesDevice     GPU adress of AllSTDPSynapsesDeviceProperties structures 
///                                   on device memory.
///  @param[in] iEdg                  Index of the Synapse to check.
///
///  @return true if there is an input spike event.
__device__ bool isSTDPSynapseSpikeQueuePostDevice(AllSTDPSynapsesDeviceProperties* allEdgesDevice, BGSIZE iEdg)
{
    uint32_t &delayQueue = allEdgesDevice->delayQueuePost_[iEdg];
    int &delayIndex = allEdgesDevice->delayIndexPost_[iEdg];
    int delayQueueLength = allEdgesDevice->delayQueuePostLength_[iEdg];

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
///  @param  allVerticesDevice       GPU address of the allNeurons struct on device memory. 
///  @param  index                  Index of the neuron to get spike history.
///  @param  offIndex               Offset of the history beffer to get.
///                                 -1 will return the last spike.
///  @param  maxSpikes              Maximum number of spikes per neuron per epoch.
///
///  @return Spike history.
__device__ uint64_t getSTDPSynapseSpikeHistoryDevice(AllSpikingNeuronsDeviceProperties* allVerticesDevice, int index, int offIndex, int maxSpikes)
{
    // offIndex is a minus offset
    int idxSp = (allVerticesDevice->spikeCount_[index] + allVerticesDevice->spikeCountOffset_[index] +  maxSpikes + offIndex) % maxSpikes;
    return allVerticesDevice->spikeHistory_[index][idxSp];
}
///@}

/******************************************
 * @name Global Functions for advanceEdges
******************************************/
///@{

///  CUDA code for advancing spiking synapses.
///  Perform updating synapses for one time step.
///
///  @param[in] totalSynapseCount  Number of synapses.
///  @param  edgeIndexMapDevice    GPU address of the EdgeIndexMap on device memory.
///  @param[in] simulationStep        The current simulation step.
///  @param[in] deltaT                Inner simulation step duration.
///  @param[in] allEdgesDevice     Pointer to AllSpikingSynapsesDeviceProperties structures 
///                                   on device memory.
__global__ void advanceSpikingSynapsesDevice ( int totalSynapseCount, EdgeIndexMap* edgeIndexMapDevice, uint64_t simulationStep, const BGFLOAT deltaT, AllSpikingSynapsesDeviceProperties* allEdgesDevice ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx >= totalSynapseCount )
                return;

                
        BGSIZE iEdg = edgeIndexMapDevice->incomingEdgeIndexMap_[idx];

        BGFLOAT &psr = allEdgesDevice->psr_[iEdg];
        BGFLOAT decay = allEdgesDevice->decay_[iEdg];

        // Checks if there is an input spike in the queue.
        bool isFired = isSpikingSynapsesSpikeQueueDevice(allEdgesDevice, iEdg);

        // is an input in the queue?
        if (isFired) {
                switch (classSynapses_d) {
                case classAllSpikingSynapses:
                       changeSpikingSynapsesPSRDevice(static_cast<AllSpikingSynapsesDeviceProperties*>(allEdgesDevice), iEdg, simulationStep, deltaT);
                        break;
                case classAllDSSynapses:
                        changeDSSynapsePSRDevice(static_cast<AllDSSynapsesDeviceProperties*>(allEdgesDevice), iEdg, simulationStep, deltaT);
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
///  @param[in] edgeIndexMapDevice    GPU address of the EdgeIndexMap on device memory.
///  @param[in] simulationStep           The current simulation step.
///  @param[in] deltaT                   Inner simulation step duration.
///  @param[in] allEdgesDevice        GPU address of AllSTDPSynapsesDeviceProperties structures 
///                                      on device memory.
///  @param[in] allVerticesDevice         GPU address of AllVertices structures on device memory.
///  @param[in] maxSpikes                Maximum number of spikes per neuron per epoch.               
__global__ void advanceSTDPSynapsesDevice ( int totalSynapseCount, EdgeIndexMap* edgeIndexMapDevice, uint64_t simulationStep, const BGFLOAT deltaT, AllSTDPSynapsesDeviceProperties* allEdgesDevice, AllSpikingNeuronsDeviceProperties* allVerticesDevice, int maxSpikes ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= totalSynapseCount )
            return;

    BGSIZE iEdg = edgeIndexMapDevice->incomingEdgeIndexMap_[idx];

    // If the synapse is inhibitory or its weight is zero, update synapse state using AllSpikingSynapses::advanceEdge method
    BGFLOAT &W = allEdgesDevice->W_[iEdg];
    if(W <= 0.0) {
        BGFLOAT &psr = allEdgesDevice->psr_[iEdg];
        BGFLOAT decay = allEdgesDevice->decay_[iEdg];

        // Checks if there is an input spike in the queue.
        bool isFired = isSpikingSynapsesSpikeQueueDevice(allEdgesDevice, iEdg);

        // is an input in the queue?
        if (isFired) {
                switch (classSynapses_d) {
                case classAllSTDPSynapses:
                        changeSpikingSynapsesPSRDevice(static_cast<AllSpikingSynapsesDeviceProperties*>(allEdgesDevice), iEdg, simulationStep, deltaT);
                        break;
                case classAllDynamicSTDPSynapses:
                	// Note: we cast void * over the allEdgesDevice, then recast it, 
                	// because AllDSSynapsesDeviceProperties inherited properties from 
                	// the AllDSSynapsesDeviceProperties and the AllSTDPSynapsesDeviceProperties.
			changeDSSynapsePSRDevice(static_cast<AllDSSynapsesDeviceProperties*>((void *)allEdgesDevice), iEdg, simulationStep, deltaT);
                        break;
                default:
                        assert(false);
                }
        }
        // decay the post spike response
        psr *= decay;
        return;
    }

    BGFLOAT &decay = allEdgesDevice->decay_[iEdg];
    BGFLOAT &psr = allEdgesDevice->psr_[iEdg];

    // is an input in the queue?
    bool fPre = isSpikingSynapsesSpikeQueueDevice(allEdgesDevice, iEdg);
    bool fPost = isSTDPSynapseSpikeQueuePostDevice(allEdgesDevice, iEdg);
    if (fPre || fPost) {
        BGFLOAT &tauspre = allEdgesDevice->tauspre_[iEdg];
        BGFLOAT &tauspost = allEdgesDevice->tauspost_[iEdg];
        BGFLOAT &taupos = allEdgesDevice->taupos_[iEdg];
        BGFLOAT &tauneg = allEdgesDevice->tauneg_[iEdg];
        int &totalDelay = allEdgesDevice->totalDelay_[iEdg];
        bool &useFroemkeDanSTDP = allEdgesDevice->useFroemkeDanSTDP_[iEdg];

        // pre and post neurons index
        int idxPre = allEdgesDevice->sourceVertexIndex_[iEdg];
        int idxPost = allEdgesDevice->destVertexIndex_[iEdg];
        int64_t spikeHistory, spikeHistory2;
        BGFLOAT delta;
        BGFLOAT epre, epost;

        if (fPre) {     // preSpikeHit
            // spikeCount points to the next available position of spike_history,
            // so the getSpikeHistory w/offset = -2 will return the spike time 
            // just one before the last spike.
            spikeHistory = getSTDPSynapseSpikeHistoryDevice(allVerticesDevice, idxPre, -2, maxSpikes);
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
                spikeHistory = getSTDPSynapseSpikeHistoryDevice(allVerticesDevice, idxPost, offIndex, maxSpikes);
                if (spikeHistory == ULONG_MAX)
                    break;
                // delta is the spike interval between pre-post spikes
                delta = -static_cast<BGFLOAT>(simulationStep - spikeHistory) * deltaT;

                if (delta <= -3.0 * tauneg)
                    break;
                if (useFroemkeDanSTDP) {
                    spikeHistory2 = getSTDPSynapseSpikeHistoryDevice(allVerticesDevice, idxPost, offIndex-1, maxSpikes);
                    if (spikeHistory2 == ULONG_MAX)
                        break;
                    epost = 1.0 - exp(-(static_cast<BGFLOAT>(spikeHistory - spikeHistory2) * deltaT) / tauspost);
                } else {
                    epost = 1.0;
                }
                stdpLearningDevice(allEdgesDevice, iEdg, delta, epost, epre);
                --offIndex;
            }

            switch (classSynapses_d) {
            case classAllSTDPSynapses:
                changeSpikingSynapsesPSRDevice(static_cast<AllSpikingSynapsesDeviceProperties*>(allEdgesDevice), iEdg, simulationStep, deltaT);
                break;
            case classAllDynamicSTDPSynapses:
                // Note: we cast void * over the allEdgesDevice, then recast it, 
                // because AllDSSynapsesDeviceProperties inherited properties from 
                // the AllDSSynapsesDeviceProperties and the AllSTDPSynapsesDeviceProperties.
                changeDSSynapsePSRDevice(static_cast<AllDSSynapsesDeviceProperties*>((void *)allEdgesDevice), iEdg, simulationStep, deltaT);
                break;
            default:
                assert(false);
            }
        }

        if (fPost) {    // postSpikeHit
            // spikeCount points to the next available position of spike_history,
            // so the getSpikeHistory w/offset = -2 will return the spike time
            // just one before the last spike.
            spikeHistory = getSTDPSynapseSpikeHistoryDevice(allVerticesDevice, idxPost, -2, maxSpikes);
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
                spikeHistory = getSTDPSynapseSpikeHistoryDevice(allVerticesDevice, idxPre, offIndex, maxSpikes);
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
                    spikeHistory2 = getSTDPSynapseSpikeHistoryDevice(allVerticesDevice, idxPre, offIndex-1, maxSpikes);
                    if (spikeHistory2 == ULONG_MAX)
                        break;
                    epre = 1.0 - exp(-(static_cast<BGFLOAT>(spikeHistory - spikeHistory2) * deltaT) / tauspre);
                } else {
                    epre = 1.0;
                }
                stdpLearningDevice(allEdgesDevice, iEdg, delta, epost, epre);
                --offIndex;
            }
        }
    }

    // decay the post spike response
    psr *= decay;
}
///@}

/******************************************
 * @name Device Functions for createEdge
******************************************/
///@{

///  Create a Spiking Synapse and connect it to the model.
///
///  @param allEdgesDevice    GPU address of the AllSpikingSynapsesDeviceProperties structures 
///                              on device memory.
///  @param neuronIndex          Index of the source neuron.
///  @param synapseOffset        Offset (into neuronIndex's) of the Synapse to create.
///  @param srcVertex            Coordinates of the source Neuron.
///  @param destVertex           Coordinates of the destination Neuron.
///  @param sumPoint             Pointer to the summation point.
///  @param deltaT               The time step size.
///  @param type                 Type of the Synapse to create.
__device__ void createSpikingSynapse(AllSpikingSynapsesDeviceProperties* allEdgesDevice, const int neuronIndex, const int synapseOffset, int sourceIndex, int destIndex, BGFLOAT *sumPoint, const BGFLOAT deltaT, edgeType type)
{
    BGFLOAT delay;
    BGSIZE maxEdges = allEdgesDevice->maxEdgesPerVertex_;
    BGSIZE iEdg = maxEdges * neuronIndex + synapseOffset;

    allEdgesDevice->inUse_[iEdg] = true;
    allEdgesDevice->destVertexIndex_[iEdg] = destIndex;
    allEdgesDevice->sourceVertexIndex_[iEdg] = sourceIndex;
    allEdgesDevice->W_[iEdg] = edgSign(type) * 10.0e-9;
    
    allEdgesDevice->delayQueue_[iEdg] = 0;
    allEdgesDevice->delayIndex_[iEdg] = 0;
    allEdgesDevice->delayQueueLength_[iEdg] = LENGTH_OF_DELAYQUEUE;

    allEdgesDevice->psr_[iEdg] = 0.0;
    allEdgesDevice->type_[iEdg] = type;

    allEdgesDevice->tau_[iEdg] = DEFAULT_tau;

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

    allEdgesDevice->tau_[iEdg] = tau;
    allEdgesDevice->decay_[iEdg] = exp( -deltaT / tau );
    allEdgesDevice->totalDelay_[iEdg] = static_cast<int>( delay / deltaT ) + 1;

    uint32_t size = allEdgesDevice->totalDelay_[iEdg] / ( sizeof(uint8_t) * 8 ) + 1;
    assert( size <= BYTES_OF_DELAYQUEUE );
}

///  Create a DS Synapse and connect it to the model.
///
///  @param allEdgesDevice    GPU address of the AllDSSynapsesDeviceProperties structures 
///                              on device memory.
///  @param neuronIndex          Index of the source neuron.
///  @param synapseOffset        Offset (into neuronIndex's) of the Synapse to create.
///  @param srcVertex            Coordinates of the source Neuron.
///  @param destVertex           Coordinates of the destination Neuron.
///  @param sumPoint             Pointer to the summation point.
///  @param deltaT               The time step size.
///  @param type                 Type of the Synapse to create.
__device__ void createDSSynapse(AllDSSynapsesDeviceProperties* allEdgesDevice, const int neuronIndex, const int synapseOffset, int sourceIndex, int destIndex, BGFLOAT *sumPoint, const BGFLOAT deltaT, edgeType type)
{
    BGFLOAT delay;
    BGSIZE maxEdges = allEdgesDevice->maxEdgesPerVertex_;
    BGSIZE iEdg = maxEdges * neuronIndex + synapseOffset;

    allEdgesDevice->inUse_[iEdg] = true;
    allEdgesDevice->destVertexIndex_[iEdg] = destIndex;
    allEdgesDevice->sourceVertexIndex_[iEdg] = sourceIndex;
    allEdgesDevice->W_[iEdg] = edgSign(type) * 10.0e-9;

    allEdgesDevice->delayQueue_[iEdg] = 0;
    allEdgesDevice->delayIndex_[iEdg] = 0;
    allEdgesDevice->delayQueueLength_[iEdg] = LENGTH_OF_DELAYQUEUE;

    allEdgesDevice->psr_[iEdg] = 0.0;
    allEdgesDevice->r_[iEdg] = 1.0;
    allEdgesDevice->u_[iEdg] = 0.4;     // DEFAULT_U
    allEdgesDevice->lastSpike_[iEdg] = ULONG_MAX;
    allEdgesDevice->type_[iEdg] = type;

    allEdgesDevice->U_[iEdg] = DEFAULT_U;
    allEdgesDevice->tau_[iEdg] = DEFAULT_tau;

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

    allEdgesDevice->U_[iEdg] = U;
    allEdgesDevice->D_[iEdg] = D;
    allEdgesDevice->F_[iEdg] = F;

    allEdgesDevice->tau_[iEdg] = tau;
    allEdgesDevice->decay_[iEdg] = exp( -deltaT / tau );
    allEdgesDevice->totalDelay_[iEdg] = static_cast<int>( delay / deltaT ) + 1;

    uint32_t size = allEdgesDevice->totalDelay_[iEdg] / ( sizeof(uint8_t) * 8 ) + 1;
    assert( size <= BYTES_OF_DELAYQUEUE );
}

///  Create a Synapse and connect it to the model.
///
///  @param allEdgesDevice    GPU address of the AllSTDPSynapsesDeviceProperties structures 
///                              on device memory.
///  @param neuronIndex          Index of the source neuron.
///  @param synapseOffset        Offset (into neuronIndex's) of the Synapse to create.
///  @param srcVertex            Coordinates of the source Neuron.
///  @param destVertex           Coordinates of the destination Neuron.
///  @param sumPoint             Pointer to the summation point.
///  @param deltaT               The time step size.
///  @param type                 Type of the Synapse to create.
__device__ void createSTDPSynapse(AllSTDPSynapsesDeviceProperties* allEdgesDevice, const int neuronIndex, const int synapseOffset, int sourceIndex, int destIndex, BGFLOAT *sumPoint, const BGFLOAT deltaT, edgeType type)
{
    BGFLOAT delay;
    BGSIZE maxEdges = allEdgesDevice->maxEdgesPerVertex_;
    BGSIZE iEdg = maxEdges * neuronIndex + synapseOffset;

    allEdgesDevice->inUse_[iEdg] = true;
    allEdgesDevice->destVertexIndex_[iEdg] = destIndex;
    allEdgesDevice->sourceVertexIndex_[iEdg] = sourceIndex;
    allEdgesDevice->W_[iEdg] = edgSign(type) * 10.0e-9;

    allEdgesDevice->delayQueue_[iEdg] = 0;
    allEdgesDevice->delayIndex_[iEdg] = 0;
    allEdgesDevice->delayQueueLength_[iEdg] = LENGTH_OF_DELAYQUEUE;

    allEdgesDevice->psr_[iEdg] = 0.0;
    allEdgesDevice->type_[iEdg] = type;

    allEdgesDevice->tau_[iEdg] = DEFAULT_tau;

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

    allEdgesDevice->tau_[iEdg] = tau;
    allEdgesDevice->decay_[iEdg] = exp( -deltaT / tau );
    allEdgesDevice->totalDelay_[iEdg] = static_cast<int>( delay / deltaT ) + 1;

    uint32_t size = allEdgesDevice->totalDelay_[iEdg] / ( sizeof(uint8_t) * 8 ) + 1;
    assert( size <= BYTES_OF_DELAYQUEUE );

    // May 1st 2020 
    // Use constants from Froemke and Dan (2002). 
    // Spike-timing-dependent synaptic modification induced by natural spike trains. Nature 416 (3/2002)
    allEdgesDevice->Apos_[iEdg] = 1.01;
    allEdgesDevice->Aneg_[iEdg] = -0.52;
    allEdgesDevice->STDPgap_[iEdg] = 2e-3;

    allEdgesDevice->totalDelayPost_[iEdg] = 0;

    allEdgesDevice->tauspost_[iEdg] = 75e-3;
    allEdgesDevice->tauspre_[iEdg] = 34e-3;

    allEdgesDevice->taupos_[iEdg] = 14.8e-3;
    allEdgesDevice->tauneg_[iEdg] = 33.8e-3;
    allEdgesDevice->Wex_[iEdg] = 5.0265e-7;

    allEdgesDevice->mupos_[iEdg] = 0;
    allEdgesDevice->muneg_[iEdg] = 0;

    allEdgesDevice->useFroemkeDanSTDP_[iEdg] = false;
}

///  Create a Synapse and connect it to the model.
///
///  @param allEdgesDevice    GPU address of the AllDynamicSTDPSynapsesDeviceProperties structures 
///                              on device memory.
///  @param neuronIndex          Index of the source neuron.
///  @param synapseOffset        Offset (into neuronIndex's) of the Synapse to create.
///  @param srcVertex            Coordinates of the source Neuron.
///  @param destVertex           Coordinates of the destination Neuron.
///  @param sumPoint             Pointer to the summation point.
///  @param deltaT               The time step size.
///  @param type                 Type of the Synapse to create.
__device__ void createDynamicSTDPSynapse(AllDynamicSTDPSynapsesDeviceProperties* allEdgesDevice, const int neuronIndex, const int synapseOffset, int sourceIndex, int destIndex, BGFLOAT *sumPoint, const BGFLOAT deltaT, edgeType type)
{
    BGFLOAT delay;
    BGSIZE maxEdges = allEdgesDevice->maxEdgesPerVertex_;
    BGSIZE iEdg = maxEdges * neuronIndex + synapseOffset;

    allEdgesDevice->inUse_[iEdg] = true;
    allEdgesDevice->destVertexIndex_[iEdg] = destIndex;
    allEdgesDevice->sourceVertexIndex_[iEdg] = sourceIndex;
    allEdgesDevice->W_[iEdg] = edgSign(type) * 10.0e-9;

    allEdgesDevice->delayQueue_[iEdg] = 0;
    allEdgesDevice->delayIndex_[iEdg] = 0;
    allEdgesDevice->delayQueueLength_[iEdg] = LENGTH_OF_DELAYQUEUE;

    allEdgesDevice->psr_[iEdg] = 0.0;
    allEdgesDevice->r_[iEdg] = 1.0;
    allEdgesDevice->u_[iEdg] = 0.4;     // DEFAULT_U
    allEdgesDevice->lastSpike_[iEdg] = ULONG_MAX;
    allEdgesDevice->type_[iEdg] = type;

    allEdgesDevice->U_[iEdg] = DEFAULT_U;
    allEdgesDevice->tau_[iEdg] = DEFAULT_tau;

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

    allEdgesDevice->U_[iEdg] = U;
    allEdgesDevice->D_[iEdg] = D;
    allEdgesDevice->F_[iEdg] = F;

    allEdgesDevice->tau_[iEdg] = tau;
    allEdgesDevice->decay_[iEdg] = exp( -deltaT / tau );
    allEdgesDevice->totalDelay_[iEdg] = static_cast<int>( delay / deltaT ) + 1;

    uint32_t size = allEdgesDevice->totalDelay_[iEdg] / ( sizeof(uint8_t) * 8 ) + 1;
    assert( size <= BYTES_OF_DELAYQUEUE );

    // May 1st 2020 
    // Use constants from Froemke and Dan (2002). 
    // Spike-timing-dependent synaptic modification induced by natural spike trains. Nature 416 (3/2002)
    allEdgesDevice->Apos_[iEdg] = 1.01;
    allEdgesDevice->Aneg_[iEdg] = -0.52;
    allEdgesDevice->STDPgap_[iEdg] = 2e-3;

    allEdgesDevice->totalDelayPost_[iEdg] = 0;

    allEdgesDevice->tauspost_[iEdg] = 75e-3;
    allEdgesDevice->tauspre_[iEdg] = 34e-3;

    allEdgesDevice->taupos_[iEdg] = 14.8e-3;
    allEdgesDevice->tauneg_[iEdg] = 33.8e-3;
    allEdgesDevice->Wex_[iEdg] = 5.0265e-7;

    allEdgesDevice->mupos_[iEdg] = 0;
    allEdgesDevice->muneg_[iEdg] = 0;

    allEdgesDevice->useFroemkeDanSTDP_[iEdg] = false;
}

/// Adds a synapse to the network.  Requires the locations of the source and
/// destination neurons.
///
/// @param allEdgesDevice      Pointer to the AllSpikingSynapsesDeviceProperties structures 
///                               on device memory.
/// @param type                   Type of the Synapse to create.
/// @param srcVertex            Coordinates of the source Neuron.
/// @param destVertex           Coordinates of the destination Neuron.
///
/// @param sumPoint              Pointer to the summation point.
/// @param deltaT                 The time step size.
/// @param W_d                    Array of synapse weight.
/// @param numVertices            The number of vertices.
__device__ void addSpikingSynapse(AllSpikingSynapsesDeviceProperties* allEdgesDevice, edgeType type, const int srcVertex, const int destVertex, int sourceIndex, int destIndex, BGFLOAT *sumPoint, const BGFLOAT deltaT, BGFLOAT* W_d, int numVertices)
{
    if (allEdgesDevice->edgeCounts_[destVertex] >= allEdgesDevice->maxEdgesPerVertex_) {
        return; // TODO: ERROR!
    }

    // add it to the list
    BGSIZE synapseIndex;
    BGSIZE maxEdges = allEdgesDevice->maxEdgesPerVertex_;
    BGSIZE synapseBegin = maxEdges * destVertex;
    for (synapseIndex = 0; synapseIndex < maxEdges; synapseIndex++) {
        if (!allEdgesDevice->inUse_[synapseBegin + synapseIndex]) {
            break;
        }
    }

    allEdgesDevice->edgeCounts_[destVertex]++;

    // create a synapse
    switch (classSynapses_d) {
    case classAllSpikingSynapses:
        createSpikingSynapse(allEdgesDevice, destVertex, synapseIndex, sourceIndex, destIndex, sumPoint, deltaT, type );
        break;
    case classAllDSSynapses:
        createDSSynapse(static_cast<AllDSSynapsesDeviceProperties *>(allEdgesDevice), destVertex, synapseIndex, sourceIndex, destIndex, sumPoint, deltaT, type );
        break;
    case classAllSTDPSynapses:
        createSTDPSynapse(static_cast<AllSTDPSynapsesDeviceProperties *>(allEdgesDevice), destVertex, synapseIndex, sourceIndex, destIndex, sumPoint, deltaT, type );
        break;
    case classAllDynamicSTDPSynapses:
        createDynamicSTDPSynapse(static_cast<AllDynamicSTDPSynapsesDeviceProperties *>(allEdgesDevice), destVertex, synapseIndex, sourceIndex, destIndex, sumPoint, deltaT, type );
        break;
    default:
        assert(false);
    }
    allEdgesDevice->W_[synapseBegin + synapseIndex] = W_d[srcVertex * numVertices + destVertex] * edgSign(type) * AllNeuroEdges::SYNAPSE_STRENGTH_ADJUSTMENT;
}

/// Remove a synapse from the network.
///
/// @param[in] allEdgesDevice      Pointer to the AllSpikingSynapsesDeviceProperties structures 
///                                   on device memory.
/// @param neuronIndex               Index of a neuron.
/// @param synapseOffset             Offset into neuronIndex's synapses.
/// @param[in] maxEdges            Maximum number of synapses per neuron.
__device__ void eraseSpikingSynapse( AllSpikingSynapsesDeviceProperties* allEdgesDevice, const int neuronIndex, const int synapseOffset, int maxEdges )
{
    BGSIZE iSync = maxEdges * neuronIndex + synapseOffset;
    allEdgesDevice->edgeCounts_[neuronIndex]--;
    allEdgesDevice->inUse_[iSync] = false;
    allEdgesDevice->W_[iSync] = 0;
}

/// Returns the type of synapse at the given coordinates
///
/// @param[in] allVerticesDevice          Pointer to the Neuron structures in device memory.
/// @param srcVertex             Index of the source neuron.
/// @param destVertex            Index of the destination neuron.
__device__ edgeType edgType( vertexType* neuronTypeMap_d, const int srcVertex, const int destVertex )
{
    if ( neuronTypeMap_d[srcVertex] == INH && neuronTypeMap_d[destVertex] == INH )
        return II;
    else if ( neuronTypeMap_d[srcVertex] == INH && neuronTypeMap_d[destVertex] == EXC )
        return IE;
    else if ( neuronTypeMap_d[srcVertex] == EXC && neuronTypeMap_d[destVertex] == INH )
        return EI;
    else if ( neuronTypeMap_d[srcVertex] == EXC && neuronTypeMap_d[destVertex] == EXC )
        return EE;

    return ETYPE_UNDEF;

}
///@}

/******************************************
 * @name Global Functions for updateSynapses
******************************************/
///@{

/// Adjust the strength of the synapse or remove it from the synapse map if it has gone below
/// zero.
///
/// @param[in] numVertices        Number of vertices.
/// @param[in] deltaT             The time step size.
/// @param[in] W_d                Array of synapse weight.
/// @param[in] maxEdges        Maximum number of synapses per neuron.
/// @param[in] allVerticesDevice   Pointer to the Neuron structures in device memory.
/// @param[in] allEdgesDevice  Pointer to the Synapse structures in device memory.
__global__ void updateSynapsesWeightsDevice( int numVertices, BGFLOAT deltaT, BGFLOAT* W_d, int maxEdges, AllSpikingNeuronsDeviceProperties* allVerticesDevice, AllSpikingSynapsesDeviceProperties* allEdgesDevice, vertexType* neuronTypeMap_d )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= numVertices )
        return;

    int adjusted = 0;
    //int could_have_been_removed = 0; // TODO: use this value
    int removed = 0;
    int added = 0;

    // Scale and add sign to the areas
    // visit each neuron 'a'
    int destVertex = idx;

    // and each destination neuron 'b'
    for (int srcVertex = 0; srcVertex < numVertices; srcVertex++) {
        // visit each synapse at (xa,ya)
        bool connected = false;
        edgeType type = edgType(neuronTypeMap_d, srcVertex, destVertex);

        // for each existing synapse
        BGSIZE existing_synapses = allEdgesDevice->edgeCounts_[destVertex];
        int existingSynapsesChecked = 0;
        for (BGSIZE synapseIndex = 0; (existingSynapsesChecked < existing_synapses) && !connected; synapseIndex++) {
            BGSIZE iEdg = maxEdges * destVertex + synapseIndex;
            if (allEdgesDevice->inUse_[iEdg] == true) {
                // if there is a synapse between a and b
                if (allEdgesDevice->sourceVertexIndex_[iEdg] == srcVertex) {
                    connected = true;
                    adjusted++;

                    // adjust the strength of the synapse or remove
                    // it from the synapse map if it has gone below
                    // zero.
                    if (W_d[srcVertex * numVertices + destVertex] <= 0) {
                        removed++;
                        eraseSpikingSynapse(allEdgesDevice, destVertex, synapseIndex, maxEdges);
                    } else {
                        // adjust
                        // g_synapseStrengthAdjustmentConstant is 1.0e-8;
                        allEdgesDevice->W_[iEdg] = W_d[srcVertex * numVertices
                            + destVertex] * edgSign(type) * AllNeuroEdges::SYNAPSE_STRENGTH_ADJUSTMENT;
                    }
                }
                existingSynapsesChecked++;
            }
        }

        // if not connected and weight(a,b) > 0, add a new synapse from a to b
        if (!connected && (W_d[srcVertex * numVertices +  destVertex] > 0)) {
            // locate summation point
            BGFLOAT* sumPoint = &( allVerticesDevice->summationMap_[destVertex] );
            added++;

            addSpikingSynapse(allEdgesDevice, type, srcVertex, destVertex, srcVertex, destVertex, sumPoint, deltaT, W_d, numVertices);
        }
    }
}


/// Adds a synapse to the network.  Requires the locations of the source and
/// destination neurons.
///
/// @param allEdgesDevice      Pointer to the Synapse structures in device memory.
/// @param pSummationMap          Pointer to the summation point.
/// @param deltaT                 The simulation time step size.
/// @param weight                 Synapse weight.
__global__ void initSynapsesDevice( int n, AllDSSynapsesDeviceProperties* allEdgesDevice, BGFLOAT *pSummationMap, const BGFLOAT deltaT, BGFLOAT weight )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n )
        return;

    // create a synapse
    int neuronIndex = idx;
    BGFLOAT* sumPoint = &( pSummationMap[neuronIndex] );
    edgeType type = allEdgesDevice->type_[neuronIndex];
    createDSSynapse(allEdgesDevice, neuronIndex, 0, 0, neuronIndex, sumPoint, deltaT, type );
    allEdgesDevice->W_[neuronIndex] = weight * AllNeuroEdges::SYNAPSE_STRENGTH_ADJUSTMENT;
}
///@}

