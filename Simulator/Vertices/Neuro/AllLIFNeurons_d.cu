/**
 * @file AllLIFNeurons_d.cu
 * 
 * @ingroup Simulation/Vertices
 *
 * @brief
 */

#include "AllLIFNeurons.h"
#include "AllVerticesDeviceFuncs.h"

#include "Book.h"

///  Update the state of all neurons for a time step
///  Notify outgoing synapses if neuron has fired.
///
///  @param  synapses               Reference to the allSynapses struct on host memory.
///  @param  allNeuronsDevice       GPU address of the allNeuronsDeviceProperties struct 
///                                 on device memory.
///  @param  allSynapsesDevice      GPU address of the allSynapsesDeviceProperties struct 
///                                 on device memory.
///  @param  randNoise              Reference to the random noise array.
///  @param  synapseIndexMapDevice  GPU address of the SynapseIndexMap on device memory.
void AllLIFNeurons::advanceVertices( IAllSynapses &synapses, void* allNeuronsDevice, void* allSynapsesDevice, float* randNoise, SynapseIndexMap* synapseIndexMapDevice )
{
    int neuron_count = Simulator::getInstance().getTotalVertices();
    int maxSpikes = (int)((Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getMaxFiringRate()));

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;

    // Advance neurons ------------->
    advanceLIFNeuronsDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, Simulator::getInstance().getMaxSynapsesPerNeuron(), maxSpikes, Simulator::getInstance().getDeltaT(), g_simulationStep, randNoise, (AllIFNeuronsDeviceProperties *)allNeuronsDevice, (AllSpikingSynapsesDeviceProperties*)allSynapsesDevice, synapseIndexMapDevice, fAllowBackPropagation_ );
}

