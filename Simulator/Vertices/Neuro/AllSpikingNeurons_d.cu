/*
 * @file AllSpikingNeurons_d.cu
 * 
 * @ingroup Simulator/Vertices
 *
 * @brief
 */

#include "AllSpikingNeurons.h"
#include "AllSpikingSynapses.h"
#include "Book.h"

///  Copy spike history data stored in device memory to host.
///
///  @param  allNeuronsDevice   GPU address of the AllSpikingNeuronsDeviceProperties struct 
///                             on device memory.
void AllSpikingNeurons::copyDeviceSpikeHistoryToHost( AllSpikingNeuronsDeviceProperties& allNeuronsDevice ) 
{
        int numNeurons = Simulator::getInstance().getTotalVertices();
        uint64_t* pSpikeHistory[numNeurons];
        HANDLE_ERROR( cudaMemcpy ( pSpikeHistory, allNeuronsDevice.spikeHistory_, numNeurons * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );

        int maxSpikes = static_cast<int> (Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getMaxFiringRate());
        for (int i = 0; i < numNeurons; i++) {
                HANDLE_ERROR( cudaMemcpy ( spikeHistory_[i], pSpikeHistory[i],
                        maxSpikes * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
        }
}

///  Copy spike counts data stored in device memory to host.
///
///  @param  allNeuronsDevice   GPU address of the AllSpikingNeuronsDeviceProperties struct 
///                             on device memory.
void AllSpikingNeurons::copyDeviceSpikeCountsToHost( AllSpikingNeuronsDeviceProperties& allNeuronsDevice ) 
{
        int numNeurons = Simulator::getInstance().getTotalVertices();

        HANDLE_ERROR( cudaMemcpy ( spikeCount_, allNeuronsDevice.spikeCount_, numNeurons * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( spikeCountOffset_, allNeuronsDevice.spikeCountOffset_, numNeurons * sizeof( int ), cudaMemcpyDeviceToHost ) );
}

///  Clear the spike counts out of all neurons in device memory.
///  (helper function of clearNeuronSpikeCounts)
///
///  @param  allNeuronsDevice   GPU address of the AllSpikingNeuronsDeviceProperties struct 
///                             on device memory.
void AllSpikingNeurons::clearDeviceSpikeCounts( AllSpikingNeuronsDeviceProperties& allNeuronsDevice ) 
{
        int numNeurons = Simulator::getInstance().getTotalVertices();

        HANDLE_ERROR( cudaMemset( allNeuronsDevice.spikeCount_, 0, numNeurons * sizeof( int ) ) );
        HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.spikeCountOffset_, spikeCountOffset_, numNeurons * sizeof( int ), cudaMemcpyHostToDevice ) );
}

///  Set some parameters used for advanceVerticesDevice.
///  Currently we set the two member variables: m_fpPreSpikeHit_h and m_fpPostSpikeHit_h.
///  These are function pointers for PreSpikeHit and PostSpikeHit device functions
///  respectively, and these functions are called from advanceVerticesDevice device
///  function. We use this scheme because we cannot not use virtual function (Polymorphism)
///  in device functions.
///
///  @param  synapses               Reference to the allSynapses struct on host memory.
void AllSpikingNeurons::setAdvanceVerticesDeviceParams(IAllEdges &synapses)
{
    AllSpikingSynapses &spSynapses = dynamic_cast<AllSpikingSynapses&>(synapses);
    fAllowBackPropagation_ = spSynapses.allowBackPropagation();
}
