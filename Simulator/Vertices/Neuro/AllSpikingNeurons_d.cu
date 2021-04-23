/*
 * @file AllSpikingNeurons_d.cu
 * 
 * @ingroup Simulator/Vertices
 *
 * @brief A container of all spiking neuron data
 */

#include "AllSpikingNeurons.h"
#include "AllSpikingSynapses.h"
#include "Book.h"

///  Copy spike history data stored in device memory to host.
///
///  @param  allVerticesDevice   GPU address of the AllSpikingNeuronsDeviceProperties struct 
///                             on device memory.
void AllSpikingNeurons::copyDeviceSpikeHistoryToHost( AllSpikingNeuronsDeviceProperties& allVerticesDevice ) 
{
        int numVertices = Simulator::getInstance().getTotalVertices();
        uint64_t* pSpikeHistory[numVertices];
        HANDLE_ERROR( cudaMemcpy ( pSpikeHistory, allVerticesDevice.spikeHistory_, numVertices * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );

        int maxSpikes = static_cast<int> (Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getMaxFiringRate());
        for (int i = 0; i < numVertices; i++) {
                HANDLE_ERROR( cudaMemcpy ( spikeHistory_[i], pSpikeHistory[i],
                        maxSpikes * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
        }
}

///  Copy spike counts data stored in device memory to host.
///
///  @param  allVerticesDevice   GPU address of the AllSpikingNeuronsDeviceProperties struct 
///                             on device memory.
void AllSpikingNeurons::copyDeviceSpikeCountsToHost( AllSpikingNeuronsDeviceProperties& allVerticesDevice ) 
{
        int numVertices = Simulator::getInstance().getTotalVertices();

        HANDLE_ERROR( cudaMemcpy ( spikeCount_, allVerticesDevice.spikeCount_, numVertices * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( spikeCountOffset_, allVerticesDevice.spikeCountOffset_, numVertices * sizeof( int ), cudaMemcpyDeviceToHost ) );
}

///  Clear the spike counts out of all neurons in device memory.
///  (helper function of clearNeuronSpikeCounts)
///
///  @param  allVerticesDevice   GPU address of the AllSpikingNeuronsDeviceProperties struct 
///                             on device memory.
void AllSpikingNeurons::clearDeviceSpikeCounts( AllSpikingNeuronsDeviceProperties& allVerticesDevice ) 
{
        int numVertices = Simulator::getInstance().getTotalVertices();

        HANDLE_ERROR( cudaMemset( allVerticesDevice.spikeCount_, 0, numVertices * sizeof( int ) ) );
        HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.spikeCountOffset_, spikeCountOffset_, numVertices * sizeof( int ), cudaMemcpyHostToDevice ) );
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
    AllSpikingSynapses &spSynapses = dynamic_cast<AllSpikingSynapses&>(synapses);
    fAllowBackPropagation_ = spSynapses.allowBackPropagation();
}
