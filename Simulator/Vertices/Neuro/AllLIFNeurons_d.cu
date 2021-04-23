/**
 * @file AllLIFNeurons_d.cu
 * 
 * @ingroup Simulator/Vertices
 *
 * @brief A container of all LIF neuron data
 */

#include "AllLIFNeurons.h"
#include "AllVerticesDeviceFuncs.h"

#include "Book.h"

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
void AllLIFNeurons::advanceVertices( AllEdges &synapses, void* allVerticesDevice, void* allEdgesDevice, float* randNoise, EdgeIndexMap* edgeIndexMapDevice )
{
    int vertex_count = Simulator::getInstance().getTotalVertices();
    int maxSpikes = (int)((Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getMaxFiringRate()));

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( vertex_count + threadsPerBlock - 1 ) / threadsPerBlock;

    // Advance neurons ------------->
    advanceLIFNeuronsDevice <<< blocksPerGrid, threadsPerBlock >>> ( vertex_count, Simulator::getInstance().getMaxEdgesPerVertex(), maxSpikes, Simulator::getInstance().getDeltaT(), g_simulationStep, randNoise, (AllIFNeuronsDeviceProperties *)allVerticesDevice, (AllSpikingSynapsesDeviceProperties*)allEdgesDevice, edgeIndexMapDevice, fAllowBackPropagation_ );
}

