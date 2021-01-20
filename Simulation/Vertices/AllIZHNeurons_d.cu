/**
 * @file AllIZHNeurons_d.cu
 *
 * @brief
 *
 * @ingroup Simulation/Vertices
 */

#include "AllSpikingSynapses.h"
#include "AllIZHNeurons.h"
#include "AllNeuronsDeviceFuncs.h"

#include "Book.h"

///  Allocate GPU memories to store all neurons' states,
///  and copy them from host to GPU memory.
///
///  @param  allNeuronsDevice   GPU address of the AllIZHNeuronsDeviceProperties struct 
///                             on device memory.
void AllIZHNeurons::allocNeuronDeviceStruct( void** allNeuronsDevice ) {
	AllIZHNeuronsDeviceProperties allNeuronsDeviceProps;

	allocDeviceStruct( allNeuronsDeviceProps );

        HANDLE_ERROR( cudaMalloc( allNeuronsDevice, sizeof( AllIZHNeuronsDeviceProperties ) ) );
        HANDLE_ERROR( cudaMemcpy ( *allNeuronsDevice, &allNeuronsDeviceProps, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyHostToDevice ) );
}

///  Allocate GPU memories to store all neurons' states.
///  (Helper function of allocNeuronDeviceStruct)
///
///  @param  allNeuronsDevice    GPU address of the AllIZHNeuronsDeviceProperties struct on device memory.
void AllIZHNeurons::allocDeviceStruct( AllIZHNeuronsDeviceProperties &allNeuronsDevice ) {
	int count = Simulator::getInstance().getTotalNeurons();

	AllIFNeurons::allocDeviceStruct( allNeuronsDevice ); 
 
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.Aconst_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.Bconst_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.Cconst_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.Dconst_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.u_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.C3_, count * sizeof( BGFLOAT ) ) );
}

///  Delete GPU memories.
///
///  @param  allNeuronsDevice   GPU address of the AllIZHNeuronsDeviceProperties struct 
///                             on device memory.
void AllIZHNeurons::deleteNeuronDeviceStruct( void* allNeuronsDevice ) {
	AllIZHNeuronsDeviceProperties allNeuronsDeviceProps;

	HANDLE_ERROR( cudaMemcpy ( &allNeuronsDeviceProps, allNeuronsDevice, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allNeuronsDeviceProps );

	HANDLE_ERROR( cudaFree( allNeuronsDevice ) );
}

///  Delete GPU memories.
///  (Helper function of deleteNeuronDeviceStruct)
///
///  @param  allNeuronsDevice    GPU address of the AllIZHNeuronsDeviceProperties struct on device memory.
void AllIZHNeurons::deleteDeviceStruct( AllIZHNeuronsDeviceProperties& allNeuronsDevice ) {
	HANDLE_ERROR( cudaFree( allNeuronsDevice.Aconst_ ) );
	HANDLE_ERROR( cudaFree( allNeuronsDevice.Bconst_ ) );
	HANDLE_ERROR( cudaFree( allNeuronsDevice.Cconst_ ) );
	HANDLE_ERROR( cudaFree( allNeuronsDevice.Dconst_ ) );
	HANDLE_ERROR( cudaFree( allNeuronsDevice.u_ ) );
	HANDLE_ERROR( cudaFree( allNeuronsDevice.C3_ ) );

	AllIFNeurons::deleteDeviceStruct( allNeuronsDevice );
}

///  Copy all neurons' data from host to device.
///
///  @param  allNeuronsDevice   GPU address of the AllIZHNeuronsDeviceProperties struct 
///                             on device memory.
void AllIZHNeurons::copyNeuronHostToDevice( void* allNeuronsDevice ) { 
	AllIZHNeuronsDeviceProperties allNeuronsDeviceProps;

	HANDLE_ERROR( cudaMemcpy ( &allNeuronsDeviceProps, allNeuronsDevice, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
	copyHostToDevice( allNeuronsDeviceProps );
}

///  Copy all neurons' data from host to device.
///  (Helper function of copyNeuronHostToDevice)
///
///  @param  allNeuronsDevice    GPU address of the AllIZHNeuronsDeviceProperties struct on device memory.
void AllIZHNeurons::copyHostToDevice( AllIZHNeuronsDeviceProperties& allNeuronsDevice ) { 
	int count = Simulator::getInstance().getTotalNeurons();

	AllIFNeurons::copyHostToDevice( allNeuronsDevice );

	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.Aconst_, Aconst_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.Bconst_, Bconst_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.Cconst_, Cconst_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.Dconst_, Dconst_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.u_, u_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.C3_, C3_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
}

///  Copy all neurons' data from device to host.
///
///  @param  allNeuronsDevice   GPU address of the AllIZHNeuronsDeviceProperties struct 
///                             on device memory.
void AllIZHNeurons::copyNeuronDeviceToHost( void* allNeuronsDevice ) {
	AllIZHNeuronsDeviceProperties allNeuronsDeviceProps;

	HANDLE_ERROR( cudaMemcpy ( &allNeuronsDeviceProps, allNeuronsDevice, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
	copyDeviceToHost( allNeuronsDeviceProps );
}

///  Copy all neurons' data from device to host.
///  (Helper function of copyNeuronDeviceToHost)
///
///  @param  allNeuronsDevice    GPU address of the AllIZHNeuronsDeviceProperties struct on device memory.
void AllIZHNeurons::copyDeviceToHost( AllIZHNeuronsDeviceProperties& allNeuronsDevice ) {
	int count = Simulator::getInstance().getTotalNeurons();

	AllIFNeurons::copyDeviceToHost( allNeuronsDevice );

	HANDLE_ERROR( cudaMemcpy ( Aconst_, allNeuronsDevice.Aconst_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Bconst_, allNeuronsDevice.Bconst_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Cconst_, allNeuronsDevice.Cconst_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Dconst_, allNeuronsDevice.Dconst_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( u_, allNeuronsDevice.u_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( C3_, allNeuronsDevice.C3_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
}

///  Copy spike history data stored in device memory to host.
///
///  @param  allNeuronsDevice   GPU address of the AllIZHNeuronsDeviceProperties struct 
///                             on device memory.
void AllIZHNeurons::copyNeuronDeviceSpikeHistoryToHost( void* allNeuronsDevice ) {
        AllIZHNeuronsDeviceProperties allNeuronsDeviceProps;
        HANDLE_ERROR( cudaMemcpy ( &allNeuronsDeviceProps, allNeuronsDevice, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::copyDeviceSpikeHistoryToHost( allNeuronsDeviceProps );
}

///  Copy spike counts data stored in device memory to host.
///
///  @param  allNeuronsDevice   GPU address of the AllIZHNeuronsDeviceProperties struct 
///                             on device memory.
void AllIZHNeurons::copyNeuronDeviceSpikeCountsToHost( void* allNeuronsDevice )
{
        AllIZHNeuronsDeviceProperties allNeuronsDeviceProps;
        HANDLE_ERROR( cudaMemcpy ( &allNeuronsDeviceProps, allNeuronsDevice, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::copyDeviceSpikeCountsToHost( allNeuronsDeviceProps );
}

///  Clear the spike counts out of all neurons.
///
///  @param  allNeuronsDevice   GPU address of the AllIZHNeuronsDeviceProperties struct 
///                             on device memory.
void AllIZHNeurons::clearNeuronSpikeCounts( void* allNeuronsDevice )
{
        AllIZHNeuronsDeviceProperties allNeuronsDeviceProps;
        HANDLE_ERROR( cudaMemcpy ( &allNeuronsDeviceProps, allNeuronsDevice, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::clearDeviceSpikeCounts( allNeuronsDeviceProps );
}

///  Notify outgoing synapses if neuron has fired.
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
