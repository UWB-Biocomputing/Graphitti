/**
 * @file AllIFNeurons_d.cu
 *
 * @brief A container of all Integate and Fire (IF) neuron data
 *
 * @ingroup Simulator/Vertices
 */

#include "AllIFNeurons.h"
#include "Book.h"

///  Allocate GPU memories to store all neurons' states,
///  and copy them from host to GPU memory.
///
///  @param  allVerticesDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
void AllIFNeurons::allocNeuronDeviceStruct( void** allVerticesDevice ) {
	AllIFNeuronsDeviceProperties allNeurons;

	allocDeviceStruct( allNeurons );

        HANDLE_ERROR( cudaMalloc( allVerticesDevice, sizeof( AllIFNeuronsDeviceProperties ) ) );
        HANDLE_ERROR( cudaMemcpy ( *allVerticesDevice, &allNeurons, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyHostToDevice ) );
}

///  Allocate GPU memories to store all neurons' states.
///  (Helper function of allocNeuronDeviceStruct)
///
///  @param  allVerticesDevice         GPU address of the AllIFNeuronsDeviceProperties struct.
void AllIFNeurons::allocDeviceStruct( AllIFNeuronsDeviceProperties &allVerticesDevice ) {
	int count = Simulator::getInstance().getTotalVertices();
	int maxSpikes = static_cast<int> (Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getMaxFiringRate());
 
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.C1_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.C2_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.Cm_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.I0_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.Iinject_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.Inoise_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.Isyn_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.Rm_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.Tau_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.Trefract_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.Vinit_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.Vm_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.Vreset_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.Vrest_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.Vthresh_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.hasFired_, count * sizeof( bool ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.numStepsInRefractoryPeriod_, count * sizeof( int ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.spikeCount_, count * sizeof( int ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.spikeCountOffset_, count * sizeof( int ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.summationMap_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allVerticesDevice.spikeHistory_, count * sizeof( uint64_t* ) ) );
	
	uint64_t* pSpikeHistory[count];
	for (int i = 0; i < count; i++) {
		HANDLE_ERROR( cudaMalloc( ( void ** ) &pSpikeHistory[i], maxSpikes * sizeof( uint64_t ) ) );
	}
	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.spikeHistory_, pSpikeHistory,
		count * sizeof( uint64_t* ), cudaMemcpyHostToDevice ) );

	// get device summation point address
	summationMap_ = allVerticesDevice.summationMap_;
}

///  Delete GPU memories.
///
///  @param  allVerticesDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
void AllIFNeurons::deleteNeuronDeviceStruct( void* allVerticesDevice ) {
	AllIFNeuronsDeviceProperties allVerticesDeviceProps;

	HANDLE_ERROR( cudaMemcpy ( &allVerticesDeviceProps, allVerticesDevice, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allVerticesDeviceProps );

	HANDLE_ERROR( cudaFree( allVerticesDevice ) );
}

///  Delete GPU memories.
///  (Helper function of deleteNeuronDeviceStruct)
///
///  @param  allVerticesDevice         GPU address of the AllIFNeuronsDeviceProperties struct.
void AllIFNeurons::deleteDeviceStruct( AllIFNeuronsDeviceProperties& allVerticesDevice ) {
	int count = Simulator::getInstance().getTotalVertices();

	uint64_t* pSpikeHistory[count];
	HANDLE_ERROR( cudaMemcpy ( pSpikeHistory, allVerticesDevice.spikeHistory_,
		count * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
	for (int i = 0; i < count; i++) {
		HANDLE_ERROR( cudaFree( pSpikeHistory[i] ) );
	}

	HANDLE_ERROR( cudaFree( allVerticesDevice.C1_ ) );
	HANDLE_ERROR( cudaFree( allVerticesDevice.C2_ ) );
	HANDLE_ERROR( cudaFree( allVerticesDevice.Cm_ ) );
	HANDLE_ERROR( cudaFree( allVerticesDevice.I0_ ) );
	HANDLE_ERROR( cudaFree( allVerticesDevice.Iinject_ ) );
	HANDLE_ERROR( cudaFree( allVerticesDevice.Inoise_ ) );
	HANDLE_ERROR( cudaFree( allVerticesDevice.Isyn_ ) );
	HANDLE_ERROR( cudaFree( allVerticesDevice.Rm_ ) );
	HANDLE_ERROR( cudaFree( allVerticesDevice.Tau_ ) );
	HANDLE_ERROR( cudaFree( allVerticesDevice.Trefract_ ) );
	HANDLE_ERROR( cudaFree( allVerticesDevice.Vinit_ ) );
	HANDLE_ERROR( cudaFree( allVerticesDevice.Vm_ ) );
	HANDLE_ERROR( cudaFree( allVerticesDevice.Vreset_ ) );
	HANDLE_ERROR( cudaFree( allVerticesDevice.Vrest_ ) );
	HANDLE_ERROR( cudaFree( allVerticesDevice.Vthresh_ ) );
	HANDLE_ERROR( cudaFree( allVerticesDevice.hasFired_ ) );
	HANDLE_ERROR( cudaFree( allVerticesDevice.numStepsInRefractoryPeriod_ ) );
	HANDLE_ERROR( cudaFree( allVerticesDevice.spikeCount_ ) );
	HANDLE_ERROR( cudaFree( allVerticesDevice.spikeCountOffset_ ) );
	HANDLE_ERROR( cudaFree( allVerticesDevice.summationMap_ ) );
	HANDLE_ERROR( cudaFree( allVerticesDevice.spikeHistory_ ) );
}

///  Copy all neurons' data from host to device.
///
///  @param  allVerticesDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
void AllIFNeurons::copyNeuronHostToDevice( void* allVerticesDevice ) { 
	AllIFNeuronsDeviceProperties allVerticesDeviceProps;

	HANDLE_ERROR( cudaMemcpy ( &allVerticesDeviceProps, allVerticesDevice, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
	copyHostToDevice( allVerticesDeviceProps );
}

///  Copy all neurons' data from host to device.
///  (Helper function of copyNeuronHostToDevice)
///
///  @param  allVerticesDevice         GPU address of the AllIFNeuronsDeviceProperties struct.
void AllIFNeurons::copyHostToDevice( AllIFNeuronsDeviceProperties& allVerticesDevice ) { 
	int count = Simulator::getInstance().getTotalVertices();

	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.C1_, C1_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.C2_, C2_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.Cm_, Cm_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.I0_, I0_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.Iinject_, Iinject_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.Inoise_, Inoise_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.Isyn_, Isyn_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.Rm_, Rm_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.Tau_, Tau_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.Trefract_, Trefract_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.Vinit_, Vinit_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.Vm_, Vm_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.Vreset_, Vreset_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.Vrest_, Vrest_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.Vthresh_, Vthresh_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.hasFired_, hasFired_, count * sizeof( bool ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.numStepsInRefractoryPeriod_, numStepsInRefractoryPeriod_, count * sizeof( int ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.spikeCount_, spikeCount_, count * sizeof( int ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allVerticesDevice.spikeCountOffset_, spikeCountOffset_, count * sizeof( int ), cudaMemcpyHostToDevice ) );

        int maxSpikes = static_cast<int> (Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getMaxFiringRate());
        uint64_t* pSpikeHistory[count];
        HANDLE_ERROR( cudaMemcpy ( pSpikeHistory, allVerticesDevice.spikeHistory_, count * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
        for (int i = 0; i < count; i++) {
                HANDLE_ERROR( cudaMemcpy ( pSpikeHistory[i], spikeHistory_[i], maxSpikes * sizeof( uint64_t ), cudaMemcpyHostToDevice ) );
        }
}

///  Copy all neurons' data from device to host.
///
///  @param  allVerticesDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
void AllIFNeurons::copyNeuronDeviceToHost( void* allVerticesDevice ) {
	AllIFNeuronsDeviceProperties allVerticesDeviceProps;

	HANDLE_ERROR( cudaMemcpy ( &allVerticesDeviceProps, allVerticesDevice, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
	copyDeviceToHost( allVerticesDeviceProps );
}

///  Copy all neurons' data from device to host.
///  (Helper function of copyNeuronDeviceToHost)
///
///  @param  allVerticesDevice         GPU address of the AllIFNeuronsDeviceProperties struct.
void AllIFNeurons::copyDeviceToHost( AllIFNeuronsDeviceProperties& allVerticesDevice ) {
	int count = Simulator::getInstance().getTotalVertices();

	HANDLE_ERROR( cudaMemcpy ( C1_, allVerticesDevice.C1_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( C2_, allVerticesDevice.C2_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Cm_, allVerticesDevice.Cm_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( I0_, allVerticesDevice.I0_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Iinject_, allVerticesDevice.Iinject_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Inoise_, allVerticesDevice.Inoise_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Isyn_, allVerticesDevice.Isyn_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Rm_, allVerticesDevice.Rm_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Tau_, allVerticesDevice.Tau_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Trefract_, allVerticesDevice.Trefract_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Vinit_, allVerticesDevice.Vinit_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Vm_, allVerticesDevice.Vm_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Vreset_, allVerticesDevice.Vreset_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Vrest_, allVerticesDevice.Vrest_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Vthresh_, allVerticesDevice.Vthresh_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( hasFired_, allVerticesDevice.hasFired_, count * sizeof( bool ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( numStepsInRefractoryPeriod_, allVerticesDevice.numStepsInRefractoryPeriod_, count * sizeof( int ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( spikeCount_, allVerticesDevice.spikeCount_, count * sizeof( int ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( spikeCountOffset_, allVerticesDevice.spikeCountOffset_, count * sizeof( int ), cudaMemcpyDeviceToHost ) );

        int maxSpikes = static_cast<int> (Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getMaxFiringRate());
        uint64_t* pSpikeHistory[count];
        HANDLE_ERROR( cudaMemcpy ( pSpikeHistory, allVerticesDevice.spikeHistory_, count * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
        for (int i = 0; i < count; i++) {
                HANDLE_ERROR( cudaMemcpy ( spikeHistory_[i], pSpikeHistory[i], maxSpikes * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
        }
}

///  Copy spike history data stored in device memory to host.
///
///  @param  allVerticesDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
void AllIFNeurons::copyNeuronDeviceSpikeHistoryToHost( void* allVerticesDevice ) 
{        
        AllIFNeuronsDeviceProperties allVerticesDeviceProps;
        HANDLE_ERROR( cudaMemcpy ( &allVerticesDeviceProps, allVerticesDevice, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );        
        AllSpikingNeurons::copyDeviceSpikeHistoryToHost( allVerticesDeviceProps );
}

///  Copy spike counts data stored in device memory to host.
///
///  @param  allVerticesDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
void AllIFNeurons::copyNeuronDeviceSpikeCountsToHost( void* allVerticesDevice )
{
        AllIFNeuronsDeviceProperties allVerticesDeviceProps;
        HANDLE_ERROR( cudaMemcpy ( &allVerticesDeviceProps, allVerticesDevice, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::copyDeviceSpikeCountsToHost( allVerticesDeviceProps );
}

///  Clear the spike counts out of all neurons.
///
///  @param  allVerticesDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
void AllIFNeurons::clearNeuronSpikeCounts( void* allVerticesDevice )
{
        AllIFNeuronsDeviceProperties allVerticesDeviceProps;
        HANDLE_ERROR( cudaMemcpy ( &allVerticesDeviceProps, allVerticesDevice, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::clearDeviceSpikeCounts( allVerticesDeviceProps );
}


///  Update the state of all neurons for a time step
///  Notify outgoing synapses if neuron has fired.
///
///  @param  synapses               Reference to the allEdges struct on host memory.
///  @param  allVerticesDevice       GPU address of the AllIFNeuronsDeviceProperties struct 
///                                 on device memory.
///  @param  allEdgesDevice      GPU address of the allEdgesDeviceProperties struct 
///                                 on device memory.
///  @param  randNoise              Reference to the random noise array.
///  @param  edgeIndexMapDevice  GPU address of the EdgeIndexMap on device memory.
void AllIFNeurons::advanceVertices( AllEdges &synapses, void* allVerticesDevice, void* allEdgesDevice, float* randNoise, EdgeIndexMap* edgeIndexMapDevice )
{
}
