/*
 * AllSTDPSynapses_d.cu
 *
 */

#include "AllSTDPSynapses.h"
#include "AllSpikingSynapses.h"
#include "GPUSpikingModel.h"
#include "AllSynapsesDeviceFuncs.h"
#include "Book.h"

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDevice  GPU address of the AllSTDPSynapsesDeviceProperties struct 
 *                             on device memory.
 */
void AllSTDPSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice ) {
	allocSynapseDeviceStruct( allSynapsesDevice, Simulator::getInstance().getTotalNeurons(), Simulator::getInstance().getMaxSynapsesPerNeuron() );
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDevice     GPU address of the AllSTDPSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  numNeurons            Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSTDPSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, int numNeurons, int maxSynapsesPerNeuron ) {
	AllSTDPSynapsesDeviceProperties allSynapsesDeviceProps;

	allocDeviceStruct( allSynapsesDeviceProps, numNeurons, maxSynapsesPerNeuron );

	HANDLE_ERROR( cudaMalloc( allSynapsesDevice, sizeof( AllSTDPSynapsesDeviceProperties ) ) );
	HANDLE_ERROR( cudaMemcpy ( *allSynapsesDevice, &allSynapsesDeviceProps, sizeof( AllSTDPSynapsesDeviceProperties ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *  (Helper function of allocSynapseDeviceStruct)
 *
 *  @param  allSynapsesDevice     GPU address of the AllSTDPSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  numNeurons            Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSTDPSynapses::allocDeviceStruct( AllSTDPSynapsesDeviceProperties &allSynapsesDevice, int numNeurons, int maxSynapsesPerNeuron ) {
        AllSpikingSynapses::allocDeviceStruct( allSynapsesDevice, numNeurons, maxSynapsesPerNeuron );

        BGSIZE maxTotalSynapses = maxSynapsesPerNeuron * numNeurons;

        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.totalDelayPost_, maxTotalSynapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.delayQueuePost_, maxTotalSynapses * sizeof( BGSIZE ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.delayIndexPost_, maxTotalSynapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.delayQueuePost_, maxTotalSynapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.tauspost_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.tauspre_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.taupos_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.tauneg_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.STDPgap_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.Wex_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.Aneg_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.Apos_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.mupos_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.muneg_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        //HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.useFroemkeDanSTDP_, maxTotalSynapses * sizeof( bool ) ) );
}

/*
 *  Delete GPU memories.
 *
 *  @param  allSynapsesDevice  GPU address of the AllSTDPSynapsesDeviceProperties struct 
 *                             on device memory.
 */
void AllSTDPSynapses::deleteSynapseDeviceStruct( void* allSynapsesDevice ) {
	AllSTDPSynapsesDeviceProperties allSynapsesDeviceProps;

	HANDLE_ERROR( cudaMemcpy ( &allSynapsesDeviceProps, allSynapsesDevice, sizeof( AllSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allSynapsesDeviceProps );

	HANDLE_ERROR( cudaFree( allSynapsesDevice ) );
}

/*
 *  Delete GPU memories.
 *  (Helper function of deleteSynapseDeviceStruct)
 *
 *  @param  allSynapsesDevice  GPU address of the AllSTDPSynapsesDeviceProperties struct 
 *                             on device memory.
 */
void AllSTDPSynapses::deleteDeviceStruct( AllSTDPSynapsesDeviceProperties& allSynapsesDevice ) {
        HANDLE_ERROR( cudaFree( allSynapsesDevice.totalDelayPost_ ) );
        HANDLE_ERROR( cudaFree( allSynapsesDevice.delayQueuePost_ ) );
        HANDLE_ERROR( cudaFree( allSynapsesDevice.delayIndexPost_ ) );
        HANDLE_ERROR( cudaFree( allSynapsesDevice.tauspost_ ) );
        HANDLE_ERROR( cudaFree( allSynapsesDevice.tauspre_ ) );
        HANDLE_ERROR( cudaFree( allSynapsesDevice.taupos_ ) );
        HANDLE_ERROR( cudaFree( allSynapsesDevice.tauneg_ ) );
        HANDLE_ERROR( cudaFree( allSynapsesDevice.STDPgap_ ) );
        HANDLE_ERROR( cudaFree( allSynapsesDevice.Wex_ ) );
        HANDLE_ERROR( cudaFree( allSynapsesDevice.Aneg_ ) );
        HANDLE_ERROR( cudaFree( allSynapsesDevice.Apos_ ) );
        HANDLE_ERROR( cudaFree( allSynapsesDevice.mupos_ ) );
        HANDLE_ERROR( cudaFree( allSynapsesDevice.muneg_) );
        //HANDLE_ERROR( cudaFree( allSynapsesDevice.useFroemkeDanSTDP_ ) );

        AllSpikingSynapses::deleteDeviceStruct( allSynapsesDevice );
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesDevice     GPU address of the AllSTDPSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  numNeurons            Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSTDPSynapses::copySynapseHostToDevice( void* allSynapsesDevice ) { // copy everything necessary
	copySynapseHostToDevice( allSynapsesDevice, Simulator::getInstance().getTotalNeurons(), Simulator::getInstance().getMaxSynapsesPerNeuron() );	
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesDevice     GPU address of the AllSTDPSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  numNeurons            Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSTDPSynapses::copySynapseHostToDevice( void* allSynapsesDevice, int numNeurons, int maxSynapsesPerNeuron ) { // copy everything necessary
	AllSTDPSynapsesDeviceProperties allSynapsesDeviceProps;

        HANDLE_ERROR( cudaMemcpy ( &allSynapsesDeviceProps, allSynapsesDevice, sizeof( AllSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	copyHostToDevice( allSynapsesDevice, allSynapsesDeviceProps, numNeurons, maxSynapsesPerNeuron );	
}

/*
 *  Copy all synapses' data from host to device.
 *  (Helper function of copySynapseHostToDevice)
 *
 *  @param  allSynapsesDevice     GPU address of the AllSTDPSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  numNeurons            Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSTDPSynapses::copyHostToDevice( void* allSynapsesDevice, AllSTDPSynapsesDeviceProperties& allSynapsesDeviceProps, int numNeurons, int maxSynapsesPerNeuron ) { // copy everything necessary 
        AllSpikingSynapses::copyHostToDevice( allSynapsesDevice, allSynapsesDeviceProps, numNeurons, maxSynapsesPerNeuron );

        BGSIZE maxTotalSynapses = maxSynapsesPerNeuron * numNeurons;
        
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.totalDelayPost_, totalDelayPost_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.delayQueuePost_, delayQueuePost_,
                maxTotalSynapses * sizeof( uint32_t ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.delayIndexPost_, delayIndexPost_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.delayQueuePost_, delayQueuePost_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.tauspost_, tauspost_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.tauspre_, tauspre_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.taupos_, taupos_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.tauneg_, tauneg_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.STDPgap_, STDPgap_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.Wex_, Wex_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.Aneg_, Aneg_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.Apos_, Apos_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.mupos_, mupos_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.muneg_, muneg_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        //HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.useFroemkeDanSTDP_, useFroemkeDanSTDP_,
                //maxTotalSynapses * sizeof( bool ), cudaMemcpyHostToDevice ) ); 
}

/*
 *  Copy all synapses' data from device to host.
 *
 *  @param  allSynapsesDevice  GPU address of the AllSTDPSynapsesDeviceProperties struct 
 *                             on device memory.
 */
void AllSTDPSynapses::copySynapseDeviceToHost( void* allSynapsesDevice ) {
	// copy everything necessary
	AllSTDPSynapsesDeviceProperties allSynapsesDeviceProps;

        HANDLE_ERROR( cudaMemcpy ( &allSynapsesDeviceProps, allSynapsesDevice, sizeof( AllSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	copyDeviceToHost( allSynapsesDeviceProps );
}

/*
 *  Copy all synapses' data from device to host.
 *  (Helper function of copySynapseDeviceToHost)
 *
 *  @param  allSynapsesDevice     GPU address of the AllSTDPSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  numNeurons            Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSTDPSynapses::copyDeviceToHost( AllSTDPSynapsesDeviceProperties& allSynapsesDevice ) {
        AllSpikingSynapses::copyDeviceToHost( allSynapsesDevice ) ;

	int numNeurons = Simulator::getInstance().getTotalNeurons();
	BGSIZE maxTotalSynapses = Simulator::getInstance().getMaxSynapsesPerNeuron() * numNeurons;

        HANDLE_ERROR( cudaMemcpy ( delayQueuePost_, allSynapsesDevice.delayQueuePost_,
                maxTotalSynapses * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( delayIndexPost_, allSynapsesDevice.delayIndexPost_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( delayQueuePost_, allSynapsesDevice.delayQueuePost_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tauspost_, allSynapsesDevice.tauspost_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tauspre_, allSynapsesDevice.tauspre_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( taupos_, allSynapsesDevice.taupos_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tauneg_, allSynapsesDevice.tauneg_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( STDPgap_, allSynapsesDevice.STDPgap_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( Wex_, allSynapsesDevice.Wex_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( Aneg_, allSynapsesDevice.Aneg_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( Apos_, allSynapsesDevice.Apos_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( mupos_, allSynapsesDevice.mupos_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( muneg_, allSynapsesDevice.muneg_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
       // HANDLE_ERROR( cudaMemcpy ( useFroemkeDanSTDP_, allSynapsesDevice.useFroemkeDanSTDP_,
                //maxTotalSynapses * sizeof( bool ), cudaMemcpyDeviceToHost ) );
}

/*
 *  Advance all the Synapses in the simulation.
 *  Update the state of all synapses for a time step.
 *
 *  @param  allSynapsesDevice      GPU address of the AllSynapsesDeviceProperties struct 
 *                                 on device memory.
 *  @param  allNeuronsDevice       GPU address of the allNeurons struct on device memory.
 *  @param  synapseIndexMapDevice  GPU address of the SynapseIndexMap on device memory.
 */
void AllSTDPSynapses::advanceSynapses( void* allSynapsesDevice, void* allNeuronsDevice, void* synapseIndexMapDevice )
{
    int maxSpikes = (int) ((Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getMaxFiringRate()));

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( totalSynapseCount_ + threadsPerBlock - 1 ) / threadsPerBlock;
    // Advance synapses ------------->
    advanceSTDPSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( totalSynapseCount_, (SynapseIndexMap*) synapseIndexMapDevice, g_simulationStep, Simulator::getInstance().getDeltaT(), 
                                (AllSTDPSynapsesDeviceProperties*)allSynapsesDevice, (AllSpikingNeuronsDeviceProperties*)allNeuronsDevice, maxSpikes );
}

/**     
 *  Set synapse class ID defined by enumClassSynapses for the caller's Synapse class.
 *  The class ID will be set to classSynapses_d in device memory,
 *  and the classSynapses_d will be referred to call a device function for the
 *  particular synapse class.
 *  Because we cannot use virtual function (Polymorphism) in device functions,
 *  we use this scheme.
 *  Note: we used to use a function pointer; however, it caused the growth_cuda crash
 *  (see issue#137).
 */
void AllSTDPSynapses::setSynapseClassID()
{
    enumClassSynapses classSynapses_h = classAllSTDPSynapses;

    HANDLE_ERROR( cudaMemcpyToSymbol(classSynapses_d, &classSynapses_h, sizeof(enumClassSynapses)) );
}


/*
 *  Prints GPU SynapsesProps data.
 *   
 *  @param  allSynapsesDeviceProps   GPU address of the corresponding SynapsesDeviceProperties struct on device memory.
 */
void AllSTDPSynapses::printGPUSynapsesProps( void* allSynapsesDeviceProps ) const
{
    AllSTDPSynapsesDeviceProperties allSynapsesProps;

    //allocate print out data members
    BGSIZE size = maxSynapsesPerNeuron_ * countNeurons_;
    if (size != 0) {
        BGSIZE *synapseCountsPrint = new BGSIZE[countNeurons_];
        BGSIZE maxSynapsesPerNeuronPrint;
        BGSIZE totalSynapseCountPrint;
        int countNeuronsPrint;
        int *sourceNeuronIndexPrint = new int[size];
        int *destNeuronIndexPrint = new int[size];
        BGFLOAT *WPrint = new BGFLOAT[size];

        synapseType *typePrint = new synapseType[size];
        BGFLOAT *psrPrint = new BGFLOAT[size];
        bool *inUsePrint = new bool[size];

        for (BGSIZE i = 0; i < size; i++) {
            inUsePrint[i] = false;
        }

        for (int i = 0; i < countNeurons_; i++) {
            synapseCountsPrint[i] = 0;
        }

        BGFLOAT *decayPrint = new BGFLOAT[size];
        int *totalDelayPrint = new int[size];
        BGFLOAT *tauPrint = new BGFLOAT[size];

        int *totalDelayPostPrint = new int[size];
        BGFLOAT *tauspostPrint = new BGFLOAT[size];
        BGFLOAT *tausprePrint = new BGFLOAT[size];
        BGFLOAT *tauposPrint = new BGFLOAT[size];
        BGFLOAT *taunegPrint = new BGFLOAT[size];
        BGFLOAT *STDPgapPrint = new BGFLOAT[size];
        BGFLOAT *WexPrint = new BGFLOAT[size];
        BGFLOAT *AnegPrint = new BGFLOAT[size];
        BGFLOAT *AposPrint = new BGFLOAT[size];
        BGFLOAT *muposPrint = new BGFLOAT[size];
        BGFLOAT *munegPrint = new BGFLOAT[size];
        bool *useFroemkeDanSTDPPrint = new bool[size];

        // copy everything
        HANDLE_ERROR( cudaMemcpy ( &allSynapsesProps, allSynapsesDeviceProps, sizeof( AllSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( synapseCountsPrint, allSynapsesProps.synapseCounts_, countNeurons_ * sizeof( BGSIZE ), cudaMemcpyDeviceToHost ) );
        maxSynapsesPerNeuronPrint = allSynapsesProps.maxSynapsesPerNeuron_;
        totalSynapseCountPrint = allSynapsesProps.totalSynapseCount_;
        countNeuronsPrint = allSynapsesProps.countNeurons_;

        // Set countNeurons_ to 0 to avoid illegal memory deallocation
        // at AllSynapsesProps deconstructor.
        allSynapsesProps.countNeurons_ = 0;

        HANDLE_ERROR( cudaMemcpy ( sourceNeuronIndexPrint, allSynapsesProps.sourceNeuronIndex_, size * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( destNeuronIndexPrint, allSynapsesProps.destNeuronIndex_, size * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( WPrint, allSynapsesProps.W_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( typePrint, allSynapsesProps.type_, size * sizeof( synapseType ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( psrPrint, allSynapsesProps.psr_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( inUsePrint, allSynapsesProps.inUse_, size * sizeof( bool ), cudaMemcpyDeviceToHost ) );

        HANDLE_ERROR( cudaMemcpy ( decayPrint, allSynapsesProps.decay_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tauPrint, allSynapsesProps.tau_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( totalDelayPrint, allSynapsesProps.totalDelay_, size * sizeof( int ), cudaMemcpyDeviceToHost ) );

        HANDLE_ERROR( cudaMemcpy ( totalDelayPostPrint, allSynapsesProps.totalDelayPost_, size * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tauspostPrint, allSynapsesProps.tauspost_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tausprePrint, allSynapsesProps.tauspre_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tauposPrint, allSynapsesProps.taupos_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( taunegPrint, allSynapsesProps.tauneg_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( STDPgapPrint, allSynapsesProps.STDPgap_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( WexPrint, allSynapsesProps.Wex_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( AnegPrint, allSynapsesProps.Aneg_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( AposPrint, allSynapsesProps.Apos_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( muposPrint, allSynapsesProps.mupos_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( munegPrint, allSynapsesProps.muneg_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        //HANDLE_ERROR( cudaMemcpy ( useFroemkeDanSTDPPrint, allSynapsesProps.useFroemkeDanSTDP_, size * sizeof( bool ), cudaMemcpyDeviceToHost ) );

        for(int i = 0; i < maxSynapsesPerNeuron_ * countNeurons_; i++) {
            if (WPrint[i] != 0.0) {
                cout << "GPU W[" << i << "] = " << WPrint[i];
                cout << " GPU sourNeuron: " << sourceNeuronIndexPrint[i];
                cout << " GPU desNeuron: " << destNeuronIndexPrint[i];
                cout << " GPU type: " << typePrint[i];
                cout << " GPU psr: " << psrPrint[i];
                cout << " GPU in_use:" << inUsePrint[i];

                cout << " GPU decay: " << decayPrint[i];
                cout << " GPU tau: " << tauPrint[i];
                cout << " GPU total_delay: " << totalDelayPrint[i];

                cout << " GPU total_delayPost: " << totalDelayPostPrint[i];
                cout << " GPU tauspost_: " << tauspostPrint[i];
                cout << " GPU tauspre_: " << tausprePrint[i];
                cout << " GPU taupos_: " << tauposPrint[i];
                cout << " GPU tauneg_: " << taunegPrint[i];
                cout << " GPU STDPgap_: " << STDPgapPrint[i];
                cout << " GPU Wex_: " << WexPrint[i];
                cout << " GPU Aneg_: " << AnegPrint[i];
                cout << " GPU Apos_: " << AposPrint[i];
                cout << " GPU mupos_: " << muposPrint[i];
                cout << " GPU muneg_: " << munegPrint[i];
               // cout << " GPU useFroemkeDanSTDP_: " << useFroemkeDanSTDPPrint[i] << endl;
            }
        }

        for (int i = 0; i < countNeurons_; i++) {
            cout << "GPU synapse_counts:" << "neuron[" << i  << "]" << synapseCountsPrint[i] << endl;
        }

        cout << "GPU totalSynapseCount:" << totalSynapseCountPrint << endl;
        cout << "GPU maxSynapsesPerNeuron:" << maxSynapsesPerNeuronPrint << endl;
        cout << "GPU countNeurons_:" << countNeuronsPrint << endl;

        // Set countNeurons_ to 0 to avoid illegal memory deallocation
        // at AllDSSynapsesProps deconstructor.
        allSynapsesProps.countNeurons_ = 0;

        delete[] destNeuronIndexPrint;
        delete[] WPrint;
        delete[] sourceNeuronIndexPrint;
        delete[] psrPrint;
        delete[] typePrint;
        delete[] inUsePrint;
        delete[] synapseCountsPrint;
        destNeuronIndexPrint = NULL;
        WPrint = NULL;
        sourceNeuronIndexPrint = NULL;
        psrPrint = NULL;
        typePrint = NULL;
        inUsePrint = NULL;
        synapseCountsPrint = NULL;

        delete[] decayPrint;
        delete[] totalDelayPrint;
        delete[] tauPrint;
        decayPrint = NULL;
        totalDelayPrint = NULL;
        tauPrint = NULL;

        delete[] totalDelayPostPrint;
        delete[] tauspostPrint;
        delete[] tausprePrint;
        delete[] tauposPrint;
        delete[] taunegPrint;
        delete[] STDPgapPrint;
        delete[] WexPrint;
        delete[] AnegPrint;
        delete[] AposPrint;
        delete[] muposPrint;
        delete[] munegPrint;
        delete[] useFroemkeDanSTDPPrint;
        totalDelayPostPrint = NULL;
        tauspostPrint = NULL;
        tausprePrint = NULL;
        tauposPrint = NULL;
        taunegPrint = NULL;
        STDPgapPrint = NULL;
        WexPrint = NULL;
        AnegPrint = NULL;
        AposPrint = NULL;
        muposPrint = NULL;
        munegPrint = NULL;
        useFroemkeDanSTDPPrint = NULL;
    }

}
