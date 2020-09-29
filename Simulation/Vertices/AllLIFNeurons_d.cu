#include "AllLIFNeurons.h"
#include "AllNeuronsDeviceFuncs.h"

#include "Book.h"

/*
 *  Update the state of all neurons for a time step
 *  Notify outgoing synapses if neuron has fired.
 *
 *  @param  synapses               Reference to the allSynapses struct on host memory.
 *  @param  allNeuronsDevice       Reference to the allNeuronsDeviceProperties struct 
 *                                 on device memory.
 *  @param  allSynapsesDevice      Reference to the allSynapsesDeviceProperties struct 
 *                                 on device memory.
 *  @param  randNoise              Reference to the random noise array.
 *  @param  synapseIndexMapDevice  Reference to the SynapseIndexMap on device memory.
 */
void AllLIFNeurons::advanceNeurons( IAllSynapses &synapses, void* allNeuronsDevice, void* allSynapsesDevice, float* randNoise, SynapseIndexMap* synapseIndexMapDevice )
{
    int neuron_count = Simulator::getInstance().getTotalNeurons();
    int maxSpikes = (int)((Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getMaxFiringRate()));

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;

    // Advance neurons ------------->
    advanceLIFNeuronsDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, Simulator::getInstance().getMaxSynapsesPerNeuron(), maxSpikes, Simulator::getInstance().getDeltaT(), g_simulationStep, randNoise, (AllIFNeuronsDeviceProperties *)allNeuronsDevice, (AllSpikingSynapsesDeviceProperties*)allSynapsesDevice, synapseIndexMapDevice, fAllowBackPropagation_ );
}

