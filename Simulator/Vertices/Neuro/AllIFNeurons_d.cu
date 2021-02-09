/**
 * @file AllIFNeurons_d.cu
 *
 * @brief
 *
 * @ingroup Simulator/Vertices
 */

#include "AllIFNeurons.h"
#include "Book.h"

///  Allocate GPU memories to store all neurons' states,
///  and copy them from host to GPU memory.
///
///  @param  allNeuronsDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
void AllIFNeurons::allocNeuronDeviceStruct( void** allNeuronsDevice ) {
	AllIFNeuronsDeviceProperties allNeurons;

	allocDeviceStruct( allNeurons );

        HANDLE_ERROR( cudaMalloc( allNeuronsDevice, sizeof( AllIFNeuronsDeviceProperties ) ) );
        HANDLE_ERROR( cudaMemcpy ( *allNeuronsDevice, &allNeurons, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyHostToDevice ) );
}

///  Allocate GPU memories to store all neurons' states.
///  (Helper function of allocNeuronDeviceStruct)
///
///  @param  allNeuronsDevice         GPU address of the AllIFNeuronsDeviceProperties struct.
void AllIFNeurons::allocDeviceStruct( AllIFNeuronsDeviceProperties &allNeuronsDevice ) {
	int count = Simulator::getInstance().getTotalVertices();
	int maxSpikes = static_cast<int> (Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getMaxFiringRate());
 
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.C1_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.C2_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.Cm_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.I0_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.Iinject_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.Inoise_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.Isyn_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.Rm_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.Tau_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.Trefract_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.Vinit_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.Vm_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.Vreset_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.Vrest_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.Vthresh_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.hasFired_, count * sizeof( bool ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.numStepsInRefractoryPeriod_, count * sizeof( int ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.spikeCount_, count * sizeof( int ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.spikeCountOffset_, count * sizeof( int ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.summationMap_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice.spikeHistory_, count * sizeof( uint64_t* ) ) );
	
	uint64_t* pSpikeHistory[count];
	for (int i = 0; i < count; i++) {
		HANDLE_ERROR( cudaMalloc( ( void ** ) &pSpikeHistory[i], maxSpikes * sizeof( uint64_t ) ) );
	}
	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.spikeHistory_, pSpikeHistory,
		count * sizeof( uint64_t* ), cudaMemcpyHostToDevice ) );

	// get device summation point address
	summationMap_ = allNeuronsDevice.summationMap_;
}

///  Delete GPU memories.
///
///  @param  allNeuronsDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
void AllIFNeurons::deleteNeuronDeviceStruct( void* allNeuronsDevice ) {
	AllIFNeuronsDeviceProperties allNeuronsDeviceProps;

	HANDLE_ERROR( cudaMemcpy ( &allNeuronsDeviceProps, allNeuronsDevice, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allNeuronsDeviceProps );

	HANDLE_ERROR( cudaFree( allNeuronsDevice ) );
}

///  Delete GPU memories.
///  (Helper function of deleteNeuronDeviceStruct)
///
///  @param  allNeuronsDevice         GPU address of the AllIFNeuronsDeviceProperties struct.
void AllIFNeurons::deleteDeviceStruct( AllIFNeuronsDeviceProperties& allNeuronsDevice ) {
	int count = Simulator::getInstance().getTotalVertices();

	uint64_t* pSpikeHistory[count];
	HANDLE_ERROR( cudaMemcpy ( pSpikeHistory, allNeuronsDevice.spikeHistory_,
		count * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
	for (int i = 0; i < count; i++) {
		HANDLE_ERROR( cudaFree( pSpikeHistory[i] ) );
	}

	HANDLE_ERROR( cudaFree( allNeuronsDevice.C1_ ) );
	HANDLE_ERROR( cudaFree( allNeuronsDevice.C2_ ) );
	HANDLE_ERROR( cudaFree( allNeuronsDevice.Cm_ ) );
	HANDLE_ERROR( cudaFree( allNeuronsDevice.I0_ ) );
	HANDLE_ERROR( cudaFree( allNeuronsDevice.Iinject_ ) );
	HANDLE_ERROR( cudaFree( allNeuronsDevice.Inoise_ ) );
	HANDLE_ERROR( cudaFree( allNeuronsDevice.Isyn_ ) );
	HANDLE_ERROR( cudaFree( allNeuronsDevice.Rm_ ) );
	HANDLE_ERROR( cudaFree( allNeuronsDevice.Tau_ ) );
	HANDLE_ERROR( cudaFree( allNeuronsDevice.Trefract_ ) );
	HANDLE_ERROR( cudaFree( allNeuronsDevice.Vinit_ ) );
	HANDLE_ERROR( cudaFree( allNeuronsDevice.Vm_ ) );
	HANDLE_ERROR( cudaFree( allNeuronsDevice.Vreset_ ) );
	HANDLE_ERROR( cudaFree( allNeuronsDevice.Vrest_ ) );
	HANDLE_ERROR( cudaFree( allNeuronsDevice.Vthresh_ ) );
	HANDLE_ERROR( cudaFree( allNeuronsDevice.hasFired_ ) );
	HANDLE_ERROR( cudaFree( allNeuronsDevice.numStepsInRefractoryPeriod_ ) );
	HANDLE_ERROR( cudaFree( allNeuronsDevice.spikeCount_ ) );
	HANDLE_ERROR( cudaFree( allNeuronsDevice.spikeCountOffset_ ) );
	HANDLE_ERROR( cudaFree( allNeuronsDevice.summationMap_ ) );
	HANDLE_ERROR( cudaFree( allNeuronsDevice.spikeHistory_ ) );
}

///  Copy all neurons' data from host to device.
///
///  @param  allNeuronsDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
void AllIFNeurons::copyNeuronHostToDevice( void* allNeuronsDevice ) { 
	AllIFNeuronsDeviceProperties allNeuronsDeviceProps;

	HANDLE_ERROR( cudaMemcpy ( &allNeuronsDeviceProps, allNeuronsDevice, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
	copyHostToDevice( allNeuronsDeviceProps );
}

///  Copy all neurons' data from host to device.
///  (Helper function of copyNeuronHostToDevice)
///
///  @param  allNeuronsDevice         GPU address of the AllIFNeuronsDeviceProperties struct.
void AllIFNeurons::copyHostToDevice( AllIFNeuronsDeviceProperties& allNeuronsDevice ) { 
	int count = Simulator::getInstance().getTotalVertices();

	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.C1_, C1_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.C2_, C2_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.Cm_, Cm_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.I0_, I0_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.Iinject_, Iinject_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.Inoise_, Inoise_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.Isyn_, Isyn_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.Rm_, Rm_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.Tau_, Tau_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.Trefract_, Trefract_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.Vinit_, Vinit_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.Vm_, Vm_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.Vreset_, Vreset_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.Vrest_, Vrest_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.Vthresh_, Vthresh_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.hasFired_, hasFired_, count * sizeof( bool ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.numStepsInRefractoryPeriod_, numStepsInRefractoryPeriod_, count * sizeof( int ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.spikeCount_, spikeCount_, count * sizeof( int ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice.spikeCountOffset_, spikeCountOffset_, count * sizeof( int ), cudaMemcpyHostToDevice ) );

        int maxSpikes = static_cast<int> (Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getMaxFiringRate());
        uint64_t* pSpikeHistory[count];
        HANDLE_ERROR( cudaMemcpy ( pSpikeHistory, allNeuronsDevice.spikeHistory_, count * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
        for (int i = 0; i < count; i++) {
                HANDLE_ERROR( cudaMemcpy ( pSpikeHistory[i], spikeHistory_[i], maxSpikes * sizeof( uint64_t ), cudaMemcpyHostToDevice ) );
        }
}

///  Copy all neurons' data from device to host.
///
///  @param  allNeuronsDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
void AllIFNeurons::copyNeuronDeviceToHost( void* allNeuronsDevice ) {
	AllIFNeuronsDeviceProperties allNeuronsDeviceProps;

	HANDLE_ERROR( cudaMemcpy ( &allNeuronsDeviceProps, allNeuronsDevice, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
	copyDeviceToHost( allNeuronsDeviceProps );
}

///  Copy all neurons' data from device to host.
///  (Helper function of copyNeuronDeviceToHost)
///
///  @param  allNeuronsDevice         GPU address of the AllIFNeuronsDeviceProperties struct.
void AllIFNeurons::copyDeviceToHost( AllIFNeuronsDeviceProperties& allNeuronsDevice ) {
	int count = Simulator::getInstance().getTotalVertices();

	HANDLE_ERROR( cudaMemcpy ( C1_, allNeuronsDevice.C1_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( C2_, allNeuronsDevice.C2_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Cm_, allNeuronsDevice.Cm_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( I0_, allNeuronsDevice.I0_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Iinject_, allNeuronsDevice.Iinject_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Inoise_, allNeuronsDevice.Inoise_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Isyn_, allNeuronsDevice.Isyn_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Rm_, allNeuronsDevice.Rm_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Tau_, allNeuronsDevice.Tau_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Trefract_, allNeuronsDevice.Trefract_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Vinit_, allNeuronsDevice.Vinit_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Vm_, allNeuronsDevice.Vm_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Vreset_, allNeuronsDevice.Vreset_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Vrest_, allNeuronsDevice.Vrest_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Vthresh_, allNeuronsDevice.Vthresh_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( hasFired_, allNeuronsDevice.hasFired_, count * sizeof( bool ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( numStepsInRefractoryPeriod_, allNeuronsDevice.numStepsInRefractoryPeriod_, count * sizeof( int ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( spikeCount_, allNeuronsDevice.spikeCount_, count * sizeof( int ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( spikeCountOffset_, allNeuronsDevice.spikeCountOffset_, count * sizeof( int ), cudaMemcpyDeviceToHost ) );

        int maxSpikes = static_cast<int> (Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getMaxFiringRate());
        uint64_t* pSpikeHistory[count];
        HANDLE_ERROR( cudaMemcpy ( pSpikeHistory, allNeuronsDevice.spikeHistory_, count * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
        for (int i = 0; i < count; i++) {
                HANDLE_ERROR( cudaMemcpy ( spikeHistory_[i], pSpikeHistory[i], maxSpikes * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
        }
}

///  Copy spike history data stored in device memory to host.
///
///  @param  allNeuronsDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
void AllIFNeurons::copyNeuronDeviceSpikeHistoryToHost( void* allNeuronsDevice ) 
{        
        AllIFNeuronsDeviceProperties allNeuronsDeviceProps;
        HANDLE_ERROR( cudaMemcpy ( &allNeuronsDeviceProps, allNeuronsDevice, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );        
        AllSpikingNeurons::copyDeviceSpikeHistoryToHost( allNeuronsDeviceProps );
}

///  Copy spike counts data stored in device memory to host.
///
///  @param  allNeuronsDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
void AllIFNeurons::copyNeuronDeviceSpikeCountsToHost( void* allNeuronsDevice )
{
        AllIFNeuronsDeviceProperties allNeuronsDeviceProps;
        HANDLE_ERROR( cudaMemcpy ( &allNeuronsDeviceProps, allNeuronsDevice, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::copyDeviceSpikeCountsToHost( allNeuronsDeviceProps );
}

///  Clear the spike counts out of all neurons.
///
///  @param  allNeuronsDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
void AllIFNeurons::clearNeuronSpikeCounts( void* allNeuronsDevice )
{
        AllIFNeuronsDeviceProperties allNeuronsDeviceProps;
        HANDLE_ERROR( cudaMemcpy ( &allNeuronsDeviceProps, allNeuronsDevice, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::clearDeviceSpikeCounts( allNeuronsDeviceProps );
}


///  Update the state of all neurons for a time step
///  Notify outgoing synapses if neuron has fired.
///
///  @param  synapses               Reference to the allSynapses struct on host memory.
///  @param  allNeuronsDevice       GPU address of the AllIFNeuronsDeviceProperties struct 
///                                 on device memory.
///  @param  allSynapsesDevice      GPU address of the allSynapsesDeviceProperties struct 
///                                 on device memory.
///  @param  randNoise              Reference to the random noise array.
///  @param  synapseIndexMapDevice  GPU address of the EdgeIndexMap on device memory.
void AllIFNeurons::advanceVertices( IAllEdges &synapses, void* allNeuronsDevice, void* allSynapsesDevice, float* randNoise, EdgeIndexMap* synapseIndexMapDevice )
{
}
