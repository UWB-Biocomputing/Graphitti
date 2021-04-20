/**
 * @file AllIZHNeurons_d.cu
 *
 * @brief A container of all Izhikevich neuron data
 *
 * @ingroup Simulator/Vertices
 */

#include "AllSpikingSynapses.h"
#include "AllIZHNeurons.h"
#include "AllVerticesDeviceFuncs.h"

#include "Book.h"

///  Allocate GPU memories to store all neurons' states,
///  and copy them from host to GPU memory.
///
///  @param  allVerticesDevice   GPU address of the AllIZHNeuronsDeviceProperties struct 
///                             on device memory.
void AllIZHNeurons::allocNeuronDeviceStruct( void** allVerticesDevice ) {
	AllIZHNeuronsDeviceProperties allVerticesDeviceProps;

	allocDeviceStruct( allVerticesDeviceProps );

        HANDLE_ERROR( cudaMalloc( allVerticesDevice, sizeof( AllIZHNeuronsDeviceProperties ) ) );
        HANDLE_ERROR( cudaMemcpy ( *allVerticesDevice, &allVerticesDeviceProps, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyHostToDevice ) );
}

///  Allocate GPU memories to store all neurons' states.
///  (Helper function of allocNeuronDeviceStruct)
///
///  @param  allVerticesDevice    GPU address of the AllIZHNeuronsDeviceProperties struct on device memory.
void AllIZHNeurons::allocDeviceStruct( AllIZHNeuronsDeviceProperties &allVerticesDevice ) {
	int count = Simulator::getInstance().getTotalVertices();

	AllIFNeurons::allocDeviceStruct( allVerticesDevice ); 
 
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.Aconst_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.Bconst_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.Cconst_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.Dconst_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.u_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.C3_, count * sizeof( BGFLOAT ) ) );
}

///  Delete GPU memories.
///
///  @param  allVerticesDevice   GPU address of the AllIZHNeuronsDeviceProperties struct 
///                             on device memory.
void AllIZHNeurons::deleteNeuronDeviceStruct( void* allVerticesDevice ) {
	AllIZHNeuronsDeviceProperties allVerticesDeviceProps;

	HANDLE_ERROR( cudaMemcpy ( &allVerticesDeviceProps, allVerticesDevice, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allVerticesDeviceProps );

	HANDLE_ERROR( cudaFree( allVerticesDevice ) );
}

///  Delete GPU memories.
///  (Helper function of deleteNeuronDeviceStruct)
///
///  @param  allVerticesDevice    GPU address of the AllIZHNeuronsDeviceProperties struct on device memory.
void AllIZHNeurons::deleteDeviceStruct( AllIZHNeuronsDeviceProperties& allVerticesDevice ) {
	HANDLE_ERROR( cudaFree( allVerticesDevice.Aconst_ ) );
	HANDLE_ERROR( cudaFree( allVerticesDevice.Bconst_ ) );
	HANDLE_ERROR( cudaFree( allVerticesDevice.Cconst_ ) );
	HANDLE_ERROR( cudaFree( allVerticesDevice.Dconst_ ) );
	HANDLE_ERROR( cudaFree( allVerticesDevice.u_ ) );
	HANDLE_ERROR( cudaFree( allVerticesDevice.C3_ ) );

	AllIFNeurons::deleteDeviceStruct( allVerticesDevice );
}

///  Copy all neurons' data from host to device.
///
///  @param  allVerticesDevice   GPU address of the AllIZHNeuronsDeviceProperties struct 
///                             on device memory.
void AllIZHNeurons::copyNeuronHostToDevice( void* allVerticesDevice ) { 
	AllIZHNeuronsDeviceProperties allVerticesDeviceProps;

	HANDLE_ERROR( cudaMemcpy ( &allVerticesDeviceProps, allVerticesDevice, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
	copyHostToDevice( allVerticesDeviceProps );
}

///  Copy all neurons' data from host to device.
///  (Helper function of copyNeuronHostToDevice)
///
///  @param  allVerticesDevice    GPU address of the AllIZHNeuronsDeviceProperties struct on device memory.
void AllIZHNeurons::copyHostToDevice( AllIZHNeuronsDeviceProperties& allVerticesDevice ) { 
	int count = Simulator::getInstance().getTotalVertices();

	AllIFNeurons::copyHostToDevice( allVerticesDevice );

	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.Aconst_, Aconst_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.Bconst_, Bconst_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.Cconst_, Cconst_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.Dconst_, Dconst_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.u_, u_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.C3_, C3_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
}

///  Copy all neurons' data from device to host.
///
///  @param  allVerticesDevice   GPU address of the AllIZHNeuronsDeviceProperties struct 
///                             on device memory.
void AllIZHNeurons::copyNeuronDeviceToHost( void* allVerticesDevice ) {
	AllIZHNeuronsDeviceProperties allVerticesDeviceProps;

	HANDLE_ERROR( cudaMemcpy ( &allVerticesDeviceProps, allVerticesDevice, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
	copyDeviceToHost( allVerticesDeviceProps );
}

///  Copy all neurons' data from device to host.
///  (Helper function of copyNeuronDeviceToHost)
///
///  @param  allVerticesDevice    GPU address of the AllIZHNeuronsDeviceProperties struct on device memory.
void AllIZHNeurons::copyDeviceToHost( AllIZHNeuronsDeviceProperties& allVerticesDevice ) {
	int count = Simulator::getInstance().getTotalVertices();

	AllIFNeurons::copyDeviceToHost( allVerticesDevice );

	HANDLE_ERROR( cudaMemcpy ( Aconst_, allVerticesDevice.Aconst_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Bconst_, allVerticesDevice.Bconst_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Cconst_, allVerticesDevice.Cconst_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Dconst_, allVerticesDevice.Dconst_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( u_, allVerticesDevice.u_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( C3_, allVerticesDevice.C3_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
}

///  Copy spike history data stored in device memory to host.
///
///  @param  allVerticesDevice   GPU address of the AllIZHNeuronsDeviceProperties struct 
///                             on device memory.
void AllIZHNeurons::copyNeuronDeviceSpikeHistoryToHost( void* allVerticesDevice ) {
        AllIZHNeuronsDeviceProperties allVerticesDeviceProps;
        HANDLE_ERROR( cudaMemcpy ( &allVerticesDeviceProps, allVerticesDevice, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::copyDeviceSpikeHistoryToHost( allVerticesDeviceProps );
}

///  Copy spike counts data stored in device memory to host.
///
///  @param  allVerticesDevice   GPU address of the AllIZHNeuronsDeviceProperties struct 
///                             on device memory.
void AllIZHNeurons::copyNeuronDeviceSpikeCountsToHost( void* allVerticesDevice )
{
        AllIZHNeuronsDeviceProperties allVerticesDeviceProps;
        HANDLE_ERROR( cudaMemcpy ( &allVerticesDeviceProps, allVerticesDevice, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::copyDeviceSpikeCountsToHost( allVerticesDeviceProps );
}

///  Clear the spike counts out of all neurons.
///
///  @param  allVerticesDevice   GPU address of the AllIZHNeuronsDeviceProperties struct 
///                             on device memory.
void AllIZHNeurons::clearNeuronSpikeCounts( void* allVerticesDevice )
{
        AllIZHNeuronsDeviceProperties allVerticesDeviceProps;
        HANDLE_ERROR( cudaMemcpy ( &allVerticesDeviceProps, allVerticesDevice, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::clearDeviceSpikeCounts( allVerticesDeviceProps );
}

///  Notify outgoing synapses if neuron has fired.
void AllIZHNeurons::advanceVertices( AllEdges &synapses, void* allVerticesDevice, void* allEdgesDevice, float* randNoise, EdgeIndexMap* edgeIndexMapDevice)
{
    int vertex_count = Simulator::getInstance().getTotalVertices();
    int maxSpikes = (int)((Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getMaxFiringRate()));

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( vertex_count + threadsPerBlock - 1 ) / threadsPerBlock;

    // Advance neurons ------------->
    advanceIZHNeuronsDevice <<< blocksPerGrid, threadsPerBlock >>> ( vertex_count, Simulator::getInstance().getMaxEdgesPerVertex(), maxSpikes, Simulator::getInstance().getDeltaT(), g_simulationStep, randNoise, (AllIZHNeuronsDeviceProperties *)allVerticesDevice, (AllSpikingSynapsesDeviceProperties*)allEdgesDevice, edgeIndexMapDevice, fAllowBackPropagation_ );
}
