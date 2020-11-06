/*
 * AllIFNeurons_d.cu
 *
 */

#include "AllIFNeurons.h"
#include "Book.h"

/*
 *  Allocate GPU memories to store all neurons' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allNeuronsDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
 */
void AllIFNeurons::allocNeuronDeviceStruct( void** allNeuronsDevice ) {
	AllIFNeuronsDeviceProperties allNeurons;

	allocDeviceStruct( allNeurons );

        HANDLE_ERROR( cudaMalloc( allNeuronsDevice, sizeof( AllIFNeuronsDeviceProperties ) ) );
        HANDLE_ERROR( cudaMemcpy ( *allNeuronsDevice, &allNeurons, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all neurons' states.
 *  (Helper function of allocNeuronDeviceStruct)
 *
 *  @param  allNeurons         GPU address of the AllIFNeuronsDeviceProperties struct.
 */
void AllIFNeurons::allocDeviceStruct( AllIFNeuronsDeviceProperties &allNeurons ) {
	int count = Simulator::getInstance().getTotalNeurons();
	int maxSpikes = static_cast<int> (Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getMaxFiringRate());
 
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.C1_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.C2_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Cm_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.I0_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Iinject_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Inoise_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Isyn_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Rm_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Tau_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Trefract_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Vinit_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Vm_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Vreset_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Vrest_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Vthresh_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.hasFired_, count * sizeof( bool ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.numStepsInRefractoryPeriod_, count * sizeof( int ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.spikeCount_, count * sizeof( int ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.spikeCountOffset_, count * sizeof( int ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.summationMap_, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.spikeHistory_, count * sizeof( uint64_t* ) ) );
	
	uint64_t* pSpikeHistory[count];
	for (int i = 0; i < count; i++) {
		HANDLE_ERROR( cudaMalloc( ( void ** ) &pSpikeHistory[i], maxSpikes * sizeof( uint64_t ) ) );
	}
	HANDLE_ERROR( cudaMemcpy ( allNeurons.spikeHistory_, pSpikeHistory,
		count * sizeof( uint64_t* ), cudaMemcpyHostToDevice ) );

	// get device summation point address
	summationMap_ = allNeurons.summationMap_;
}

/*
 *  Delete GPU memories.
 *
 *  @param  allNeuronsDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
 */
void AllIFNeurons::deleteNeuronDeviceStruct( void* allNeuronsDevice ) {
	AllIFNeuronsDeviceProperties allNeurons;

	HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allNeurons );

	HANDLE_ERROR( cudaFree( allNeuronsDevice ) );
}

/*
 *  Delete GPU memories.
 *  (Helper function of deleteNeuronDeviceStruct)
 *
 *  @param  allNeurons         GPU address of the AllIFNeuronsDeviceProperties struct.
 */
void AllIFNeurons::deleteDeviceStruct( AllIFNeuronsDeviceProperties& allNeurons ) {
	int count = Simulator::getInstance().getTotalNeurons();

	uint64_t* pSpikeHistory[count];
	HANDLE_ERROR( cudaMemcpy ( pSpikeHistory, allNeurons.spikeHistory_,
		count * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
	for (int i = 0; i < count; i++) {
		HANDLE_ERROR( cudaFree( pSpikeHistory[i] ) );
	}

	HANDLE_ERROR( cudaFree( allNeurons.C1_ ) );
	HANDLE_ERROR( cudaFree( allNeurons.C2_ ) );
	HANDLE_ERROR( cudaFree( allNeurons.Cm_ ) );
	HANDLE_ERROR( cudaFree( allNeurons.I0_ ) );
	HANDLE_ERROR( cudaFree( allNeurons.Iinject_ ) );
	HANDLE_ERROR( cudaFree( allNeurons.Inoise_ ) );
	HANDLE_ERROR( cudaFree( allNeurons.Isyn_ ) );
	HANDLE_ERROR( cudaFree( allNeurons.Rm_ ) );
	HANDLE_ERROR( cudaFree( allNeurons.Tau_ ) );
	HANDLE_ERROR( cudaFree( allNeurons.Trefract_ ) );
	HANDLE_ERROR( cudaFree( allNeurons.Vinit_ ) );
	HANDLE_ERROR( cudaFree( allNeurons.Vm_ ) );
	HANDLE_ERROR( cudaFree( allNeurons.Vreset_ ) );
	HANDLE_ERROR( cudaFree( allNeurons.Vrest_ ) );
	HANDLE_ERROR( cudaFree( allNeurons.Vthresh_ ) );
	HANDLE_ERROR( cudaFree( allNeurons.hasFired_ ) );
	HANDLE_ERROR( cudaFree( allNeurons.numStepsInRefractoryPeriod_ ) );
	HANDLE_ERROR( cudaFree( allNeurons.spikeCount_ ) );
	HANDLE_ERROR( cudaFree( allNeurons.spikeCountOffset_ ) );
	HANDLE_ERROR( cudaFree( allNeurons.summationMap_ ) );
	HANDLE_ERROR( cudaFree( allNeurons.spikeHistory_ ) );
}

/*
 *  Copy all neurons' data from host to device.
 *
 *  @param  allNeuronsDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
 */
void AllIFNeurons::copyNeuronHostToDevice( void* allNeuronsDevice ) { 
	AllIFNeuronsDeviceProperties allNeurons;

	HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
	copyHostToDevice( allNeurons );
}

/*
 *  Copy all neurons' data from host to device.
 *  (Helper function of copyNeuronHostToDevice)
 *
 *  @param  allNeurons         GPU address of the AllIFNeuronsDeviceProperties struct.
 */
void AllIFNeurons::copyHostToDevice( AllIFNeuronsDeviceProperties& allNeurons ) { 
	int count = Simulator::getInstance().getTotalNeurons();

	HANDLE_ERROR( cudaMemcpy ( allNeurons.C1_, C1_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.C2_, C2_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Cm_, Cm_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.I0_, I0_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Iinject_, Iinject_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Inoise_, Inoise_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Isyn_, Isyn_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Rm_, Rm_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Tau_, Tau_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Trefract_, Trefract_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Vinit_, Vinit_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Vm_, Vm_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Vreset_, Vreset_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Vrest_, Vrest_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Vthresh_, Vthresh_, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.hasFired_, hasFired_, count * sizeof( bool ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.numStepsInRefractoryPeriod_, numStepsInRefractoryPeriod_, count * sizeof( int ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.spikeCount_, spikeCount_, count * sizeof( int ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.spikeCountOffset_, spikeCountOffset_, count * sizeof( int ), cudaMemcpyHostToDevice ) );

        int maxSpikes = static_cast<int> (Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getMaxFiringRate());
        uint64_t* pSpikeHistory[count];
        HANDLE_ERROR( cudaMemcpy ( pSpikeHistory, allNeurons.spikeHistory_, count * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
        for (int i = 0; i < count; i++) {
                HANDLE_ERROR( cudaMemcpy ( pSpikeHistory[i], spikeHistory_[i], maxSpikes * sizeof( uint64_t ), cudaMemcpyHostToDevice ) );
        }
}

/*
 *  Copy all neurons' data from device to host.
 *
 *  @param  allNeuronsDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
 */
void AllIFNeurons::copyNeuronDeviceToHost( void* allNeuronsDevice ) {
	AllIFNeuronsDeviceProperties allNeurons;

	HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
	copyDeviceToHost( allNeurons );
}

/*
 *  Copy all neurons' data from device to host.
 *  (Helper function of copyNeuronDeviceToHost)
 *
 *  @param  allNeurons         GPU address of the AllIFNeuronsDeviceProperties struct.
 */
void AllIFNeurons::copyDeviceToHost( AllIFNeuronsDeviceProperties& allNeurons ) {
	int count = Simulator::getInstance().getTotalNeurons();

	HANDLE_ERROR( cudaMemcpy ( C1_, allNeurons.C1_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( C2_, allNeurons.C2_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Cm_, allNeurons.Cm_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( I0_, allNeurons.I0_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Iinject_, allNeurons.Iinject_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Inoise_, allNeurons.Inoise_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Isyn_, allNeurons.Isyn_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Rm_, allNeurons.Rm_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Tau_, allNeurons.Tau_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Trefract_, allNeurons.Trefract_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Vinit_, allNeurons.Vinit_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Vm_, allNeurons.Vm_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Vreset_, allNeurons.Vreset_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Vrest_, allNeurons.Vrest_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Vthresh_, allNeurons.Vthresh_, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( hasFired_, allNeurons.hasFired_, count * sizeof( bool ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( numStepsInRefractoryPeriod_, allNeurons.numStepsInRefractoryPeriod_, count * sizeof( int ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( spikeCount_, allNeurons.spikeCount_, count * sizeof( int ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( spikeCountOffset_, allNeurons.spikeCountOffset_, count * sizeof( int ), cudaMemcpyDeviceToHost ) );

        int maxSpikes = static_cast<int> (Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getMaxFiringRate());
        uint64_t* pSpikeHistory[count];
        HANDLE_ERROR( cudaMemcpy ( pSpikeHistory, allNeurons.spikeHistory_, count * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
        for (int i = 0; i < count; i++) {
                HANDLE_ERROR( cudaMemcpy ( spikeHistory_[i], pSpikeHistory[i], maxSpikes * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
        }
}

/*
 *  Copy spike history data stored in device memory to host.
 *
 *  @param  allNeuronsDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
 */
void AllIFNeurons::copyNeuronDeviceSpikeHistoryToHost( void* allNeuronsDevice ) 
{        
        AllIFNeuronsDeviceProperties allNeurons;
        HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );        
        AllSpikingNeurons::copyDeviceSpikeHistoryToHost( allNeurons );
}

/*
 *  Copy spike counts data stored in device memory to host.
 *
 *  @param  allNeuronsDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
 */
void AllIFNeurons::copyNeuronDeviceSpikeCountsToHost( void* allNeuronsDevice )
{
        AllIFNeuronsDeviceProperties allNeurons;
        HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::copyDeviceSpikeCountsToHost( allNeurons );
}

/*
 *  Clear the spike counts out of all neurons.
 *
 *  @param  allNeuronsDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
 */
void AllIFNeurons::clearNeuronSpikeCounts( void* allNeuronsDevice )
{
        AllIFNeuronsDeviceProperties allNeurons;
        HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::clearDeviceSpikeCounts( allNeurons );
}


/*
 *  Update the state of all neurons for a time step
 *  Notify outgoing synapses if neuron has fired.
 *
 *  @param  synapses               Reference to the allSynapses struct on host memory.
 *  @param  allNeuronsDevice       GPU address of the AllIFNeuronsDeviceProperties struct 
 *                                 on device memory.
 *  @param  allSynapsesDevice      GPU address of the allSynapsesDeviceProperties struct 
 *                                 on device memory.
 *  @param  randNoise              Reference to the random noise array.
 *  @param  synapseIndexMapDevice  GPU address of the SynapseIndexMap on device memory.
 */
void AllIFNeurons::advanceNeurons( IAllSynapses &synapses, void* allNeuronsDevice, void* allSynapsesDevice, float* randNoise, SynapseIndexMap* synapseIndexMapDevice )
{
}
