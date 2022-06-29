/**
 * @file AllSynapsesDeviceFuncs_d_Growth.cpp
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
///#include "AllSynapsesDeviceFuncs_d.cpp"


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

