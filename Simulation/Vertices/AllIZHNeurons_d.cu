/*
 * AllIZHNeurons.cu
 *
 */

#include "AllSpikingSynapses.h"
#include "AllIZHNeurons.h"
#include "AllNeuronsDeviceFuncs.h"

#include "Book.h"

/*
 *  Allocate GPU memories to store all neurons' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allNeuronsDevice   Reference to the AllIZHNeuronsDeviceProperties struct 
 *                             on device memory.
 */
void AllIZHNeurons::allocNeuronDeviceStruct( void** allNeuronsDevice ) {
	AllIZHNeuronsDeviceProperties allNeurons;

	allocDeviceStruct( allNeurons );

        HANDLE_ERROR( cudaMalloc( allNeuronsDevice, sizeof( AllIZHNeuronsDeviceProperties ) ) );
        HANDLE_ERROR( cudaMemcpy ( *allNeuronsDevice, &allNeurons, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all neurons' states.
 *  (Helper function of allocNeuronDeviceStruct)
 *
 *  @param  allNeurons         Reference to the AllIZHNeuronsDeviceProperties struct.
 */
void AllIZHNeurons::allocDeviceStruct( AllIZHNeuronsDeviceProperties &allNeurons ) {
	int count = Simulator::getInstance().getTotalNeurons();

	AllIFNeurons::allocDeviceStruct( allNeurons ); 
 
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Aconst_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Bconst_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Cconst_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Dconst_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.u_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.C3_, count * sizeof( BGFLOAT ) ) );
}

/*
 *  Delete GPU memories.
 *
 *  @param  allNeuronsDevice   Reference to the AllIZHNeuronsDeviceProperties struct 
 *                             on device memory.
 */
void AllIZHNeurons::deleteNeuronDeviceStruct( void* allNeuronsDevice ) {
	AllIZHNeuronsDeviceProperties allNeurons;

	HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allNeurons );

	HANDLE_ERROR( cudaFree( allNeuronsDevice ) );
}

/*
 *  Delete GPU memories.
 *  (Helper function of deleteNeuronDeviceStruct)
 *
 *  @param  allNeurons         Reference to the AllIZHNeuronsDeviceProperties struct.
 */
void AllIZHNeurons::deleteDeviceStruct( AllIZHNeuronsDeviceProperties& allNeurons ) {
	HANDLE_ERROR( cudaFree( allNeurons.Aconst_ ) );
	HANDLE_ERROR( cudaFree( allNeurons.Bconst_ ) );
	HANDLE_ERROR( cudaFree( allNeurons.Cconst_ ) );
	HANDLE_ERROR( cudaFree( allNeurons.Dconst_ ) );
	HANDLE_ERROR( cudaFree( allNeurons.u_ ) );
	HANDLE_ERROR( cudaFree( allNeurons.C3_ ) );

	AllIFNeurons::deleteDeviceStruct( allNeurons );
}

/*
 *  Copy all neurons' data from host to device.
 *
 *  @param  allNeuronsDevice   Reference to the AllIZHNeuronsDeviceProperties struct 
 *                             on device memory.
 */
void AllIZHNeurons::copyNeuronHostToDevice( void* allNeuronsDevice ) { 
	AllIZHNeuronsDeviceProperties allNeurons;

	HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
	copyHostToDevice( allNeurons );
}

/*
 *  Copy all neurons' data from host to device.
 *  (Helper function of copyNeuronHostToDevice)
 *
 *  @param  allNeurons         Reference to the AllIZHNeuronsDeviceProperties struct.
 */
void AllIZHNeurons::copyHostToDevice( AllIZHNeuronsDeviceProperties& allNeurons ) { 
	int count = Simulator::getInstance().getTotalNeurons();

	AllIFNeurons::copyHostToDevice( allNeurons );

	HANDLE_ERROR( cudaMemcpy ( allNeurons.Aconst_, Aconst_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Bconst_, Bconst_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Cconst_, Cconst_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Dconst_, Dconst_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.u_, u_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.C3_, C3_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
}

/*
 *  Copy all neurons' data from device to host.
 *
 *  @param  allNeuronsDevice   Reference to the AllIZHNeuronsDeviceProperties struct 
 *                             on device memory.
 */
void AllIZHNeurons::copyNeuronDeviceToHost( void* allNeuronsDevice ) {
	AllIZHNeuronsDeviceProperties allNeurons;

	HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
	copyDeviceToHost( allNeurons );
}

/*
 *  Copy all neurons' data from device to host.
 *  (Helper function of copyNeuronDeviceToHost)
 *
 *  @param  allNeurons         Reference to the AllIZHNeuronsDeviceProperties struct.
 */
void AllIZHNeurons::copyDeviceToHost( AllIZHNeuronsDeviceProperties& allNeurons ) {
	int count = Simulator::getInstance().getTotalNeurons();

	AllIFNeurons::copyDeviceToHost( allNeurons );

	HANDLE_ERROR( cudaMemcpy ( Aconst_, allNeurons.Aconst_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Bconst_, allNeurons.Bconst_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Cconst_, allNeurons.Cconst_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Dconst_, allNeurons.Dconst_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( u_, allNeurons.u_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( C3_, allNeurons.C3_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
}

/*
 *  Copy spike history data stored in device memory to host.
 *
 *  @param  allNeuronsDevice   Reference to the AllIZHNeuronsDeviceProperties struct 
 *                             on device memory.
 */
void AllIZHNeurons::copyNeuronDeviceSpikeHistoryToHost( void* allNeuronsDevice ) {
        AllIZHNeuronsDeviceProperties allNeurons;
        HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::copyDeviceSpikeHistoryToHost( allNeurons );
}

/*
 *  Copy spike counts data stored in device memory to host.
 *
 *  @param  allNeuronsDevice   Reference to the AllIZHNeuronsDeviceProperties struct 
 *                             on device memory.
 */
void AllIZHNeurons::copyNeuronDeviceSpikeCountsToHost( void* allNeuronsDevice )
{
        AllIZHNeuronsDeviceProperties allNeurons;
        HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::copyDeviceSpikeCountsToHost( allNeurons );
}

/*
 *  Clear the spike counts out of all neurons.
 *
 *  @param  allNeuronsDevice   Reference to the AllIZHNeuronsDeviceProperties struct 
 *                             on device memory.
 */
void AllIZHNeurons::clearNeuronSpikeCounts( void* allNeuronsDevice )
{
        AllIZHNeuronsDeviceProperties allNeurons;
        HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::clearDeviceSpikeCounts( allNeurons );
}

/*
 *  Notify outgoing synapses if neuron has fired.
 *
 */
void AllIZHNeurons::advanceNeurons( IAllSynapses &synapses, void* allNeuronsDevice, void* allSynapsesDevice, float* randNoise, SynapseIndexMap* synapseIndexMapDevice)
{
    int neuron_count = Simulator::getInstance().getTotalNeurons();
    int maxSpikes = (int)((Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getMaxFiringRate()));

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;

    // Advance neurons ------------->
    advanceIZHNeuronsDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, Simulator::getInstance().getMaxSynapsesPerNeuron(), maxSpikes, Simulator::getInstance().getDeltaT(), g_simulationStep, randNoise, (AllIZHNeuronsDeviceProperties *)allNeuronsDevice, (AllSpikingSynapsesDeviceProperties*)allSynapsesDevice, synapseIndexMapDevice, fAllowBackPropagation_ );
}
