/*
 * AllSpikingSynapses.cu
 *
 */

#include "AllSpikingSynapses.h"
#include "AllSynapsesDeviceFuncs.h"
#include "Book.h"

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDevice  GPU address of the AllSpikingSynapsesDeviceProperties struct 
 *                             on device memory.
 */
void AllSpikingSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice ) {
        allocSynapseDeviceStruct( allSynapsesDevice, Simulator::getInstance().getTotalNeurons(), Simulator::getInstance().getMaxSynapsesPerNeuron() );
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDevice     GPU address of the AllSpikingSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  numNeurons            Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSpikingSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, int numNeurons, int maxSynapsesPerNeuron ) {
        AllSpikingSynapsesDeviceProperties allSynapses;

        allocDeviceStruct( allSynapses, numNeurons, maxSynapsesPerNeuron );

        HANDLE_ERROR( cudaMalloc( allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ) ) );
        HANDLE_ERROR( cudaMemcpy ( *allSynapsesDevice, &allSynapses, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *  (Helper function of allocSynapseDeviceStruct)
 *
 *  @param  allSynapsesDevice     GPU address of the AllSpikingSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  numNeurons            Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSpikingSynapses::allocDeviceStruct( AllSpikingSynapsesDeviceProperties &allSynapses, int numNeurons, int maxSynapsesPerNeuron ) {
        BGSIZE maxTotalSynapses = maxSynapsesPerNeuron * numNeurons;

        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.sourceNeuronIndex_, maxTotalSynapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.destNeuronIndex_, maxTotalSynapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.W_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.type_, maxTotalSynapses * sizeof( synapseType ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.psr_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.inUse_, maxTotalSynapses * sizeof( bool ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.synapseCounts_, numNeurons * sizeof( BGSIZE ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.decay_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.tau_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.totalDelay_, maxTotalSynapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.delayQueue_, maxTotalSynapses * sizeof( uint32_t ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.delayIndex_, maxTotalSynapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.delayQueueLength_, maxTotalSynapses * sizeof( int ) ) );
}

/*
 *  Delete GPU memories.
 *
 *  @param  allSynapsesDevice  GPU address of the AllSpikingSynapsesDeviceProperties struct 
 *                             on device memory.
 */
void AllSpikingSynapses::deleteSynapseDeviceStruct( void* allSynapsesDevice ) {
        AllSpikingSynapsesDeviceProperties allSynapses;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

        deleteDeviceStruct( allSynapses );

        HANDLE_ERROR( cudaFree( allSynapsesDevice ) );
}

/*
 *  Delete GPU memories.
 *  (Helper function of deleteSynapseDeviceStruct)
 *
 *  @param  allSynapsesDevice  GPU address of the AllSpikingSynapsesDeviceProperties struct 
 *                             on device memory.
 */
void AllSpikingSynapses::deleteDeviceStruct( AllSpikingSynapsesDeviceProperties& allSynapses ) {
        HANDLE_ERROR( cudaFree( allSynapses.sourceNeuronIndex_ ) );
        HANDLE_ERROR( cudaFree( allSynapses.destNeuronIndex_ ) );
        HANDLE_ERROR( cudaFree( allSynapses.W_ ) );
        HANDLE_ERROR( cudaFree( allSynapses.type_ ) );
        HANDLE_ERROR( cudaFree( allSynapses.psr_ ) );
        HANDLE_ERROR( cudaFree( allSynapses.inUse_ ) );
        HANDLE_ERROR( cudaFree( allSynapses.synapseCounts_ ) );
        HANDLE_ERROR( cudaFree( allSynapses.decay_ ) );
        HANDLE_ERROR( cudaFree( allSynapses.tau_ ) );
        HANDLE_ERROR( cudaFree( allSynapses.totalDelay_ ) );
        HANDLE_ERROR( cudaFree( allSynapses.delayQueue_ ) );
        HANDLE_ERROR( cudaFree( allSynapses.delayIndex_ ) );
        HANDLE_ERROR( cudaFree( allSynapses.delayQueueLength_ ) );

        // Set countNeurons_ to 0 to avoid illegal memory deallocation 
        // at AllSpikingSynapses deconstructor.
        //allSynapses.countNeurons_ = 0;
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesDevice  GPU address of the AllSpikingSynapsesDeviceProperties struct 
 *                             on device memory.
 */
void AllSpikingSynapses::copySynapseHostToDevice( void* allSynapsesDevice ) { // copy everything necessary
        copySynapseHostToDevice( allSynapsesDevice, Simulator::getInstance().getTotalNeurons(), Simulator::getInstance().getMaxSynapsesPerNeuron() );
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesDevice     GPU address of the AllSpikingSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  numNeurons            Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSpikingSynapses::copySynapseHostToDevice( void* allSynapsesDevice, int numNeurons, int maxSynapsesPerNeuron ) { // copy everything necessary
        AllSpikingSynapsesDeviceProperties allSynapses;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

        copyHostToDevice( allSynapsesDevice, allSynapses, numNeurons, maxSynapsesPerNeuron );
}

/*
 *  Copy all synapses' data from host to device.
 *  (Helper function of copySynapseHostToDevice)
 *
 *  @param  allSynapsesDevice     GPU address of the AllSpikingSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  numNeurons            Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSpikingSynapses::copyHostToDevice( void* allSynapsesDevice, AllSpikingSynapsesDeviceProperties& allSynapses, int numNeurons, int maxSynapsesPerNeuron ) { // copy everything necessary 
        BGSIZE maxTotalSynapses = maxSynapsesPerNeuron * numNeurons;

        allSynapses.maxSynapsesPerNeuron_ = maxSynapsesPerNeuron_;
        allSynapses.totalSynapseCount_ = totalSynapseCount_;
        allSynapses.countNeurons_ = countNeurons_;
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDevice, &allSynapses, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyHostToDevice ) );

        // Set countNeurons_ to 0 to avoid illegal memory deallocation 
        // at AllSpikingSynapses deconstructor.
        allSynapses.countNeurons_ = 0;

        HANDLE_ERROR( cudaMemcpy ( allSynapses.sourceNeuronIndex_, sourceNeuronIndex_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.destNeuronIndex_, destNeuronIndex_,
                maxTotalSynapses * sizeof( int ),  cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.W_, W_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.type_, type_,
                maxTotalSynapses * sizeof( synapseType ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.psr_, psr_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.inUse_, inUse_,
                maxTotalSynapses * sizeof( bool ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.synapseCounts_, synapseCounts_,
                        numNeurons * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.decay_, decay_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.tau_, tau_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.totalDelay_, totalDelay_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.delayQueue_, delayQueue_,
                maxTotalSynapses * sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.delayIndex_, delayIndex_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.delayQueueLength_, delayQueueLength_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyHostToDevice ) );
}

/*
 *  Copy all synapses' data from device to host.
 *
 *  @param  allSynapsesDevice  GPU address of the AllSpikingSynapsesDeviceProperties struct 
 *                             on device memory.
 */
void AllSpikingSynapses::copySynapseDeviceToHost( void* allSynapsesDevice ) {
        // copy everything necessary
        AllSpikingSynapsesDeviceProperties allSynapses;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

        copyDeviceToHost( allSynapses );
}

/*
 *  Copy all synapses' data from device to host.
 *  (Helper function of copySynapseDeviceToHost)
 *
 *  @param  allSynapsesDevice     GPU address of the AllSpikingSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  numNeurons            Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSpikingSynapses::copyDeviceToHost( AllSpikingSynapsesDeviceProperties& allSynapses ) {
        int numNeurons = Simulator::getInstance().getTotalNeurons();
        BGSIZE maxTotalSynapses = Simulator::getInstance().getMaxSynapsesPerNeuron() * numNeurons;

        HANDLE_ERROR( cudaMemcpy ( synapseCounts_, allSynapses.synapseCounts_,
                numNeurons * sizeof( BGSIZE ), cudaMemcpyDeviceToHost ) );
        maxSynapsesPerNeuron_ = allSynapses.maxSynapsesPerNeuron_;
        totalSynapseCount_ = allSynapses.totalSynapseCount_;
        countNeurons_ = allSynapses.countNeurons_;

        // Set countNeurons_ to 0 to avoid illegal memory deallocation 
        // at AllSpikingSynapses deconstructor.
        allSynapses.countNeurons_ = 0;

        HANDLE_ERROR( cudaMemcpy ( sourceNeuronIndex_, allSynapses.sourceNeuronIndex_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( destNeuronIndex_, allSynapses.destNeuronIndex_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( W_, allSynapses.W_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( type_, allSynapses.type_,
                maxTotalSynapses * sizeof( synapseType ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( psr_, allSynapses.psr_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( inUse_, allSynapses.inUse_,
                maxTotalSynapses * sizeof( bool ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( decay_, allSynapses.decay_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tau_, allSynapses.tau_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( totalDelay_, allSynapses.totalDelay_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( delayQueue_, allSynapses.delayQueue_,
                maxTotalSynapses * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( delayIndex_, allSynapses.delayIndex_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( delayQueueLength_, allSynapses.delayQueueLength_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
}

/*
 *  Get synapse_counts in AllSynapses struct on device memory.
 *
 *  @param  allSynapsesDevice  GPU address of the AllSpikingSynapsesDeviceProperties struct 
 *                             on device memory.
 */
void AllSpikingSynapses::copyDeviceSynapseCountsToHost( void* allSynapsesDevice )
{
        AllSpikingSynapsesDeviceProperties allSynapses;
        int neuronCount = Simulator::getInstance().getTotalNeurons();

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( synapseCounts_, allSynapses.synapseCounts_, neuronCount * sizeof( BGSIZE ), cudaMemcpyDeviceToHost ) );

        // Set countNeurons_ to 0 to avoid illegal memory deallocation 
        // at AllSpikingSynapses deconstructor.
        //allSynapses.countNeurons_ = 0;
}

/* 
 *  Get summationCoord and in_use in AllSynapses struct on device memory.
 *
 *  @param  allSynapsesDevice  GPU address of the AllSpikingSynapsesDeviceProperties struct
 *                             on device memory.
 */
void AllSpikingSynapses::copyDeviceSynapseSumIdxToHost(void* allSynapsesDevice )
{
        AllSpikingSynapsesDeviceProperties allSynapses;
        BGSIZE maxTotalSynapses = Simulator::getInstance().getMaxSynapsesPerNeuron() * Simulator::getInstance().getTotalNeurons();

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( sourceNeuronIndex_, allSynapses.sourceNeuronIndex_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( inUse_, allSynapses.inUse_,
                maxTotalSynapses * sizeof( bool ), cudaMemcpyDeviceToHost ) );
       
        // Set countNeurons_ to 0 to avoid illegal memory deallocation 
        // at AllSpikingSynapses deconstructor.
        //allSynapses.countNeurons_ = 0;
}

/*
 *  Set some parameters used for advanceSynapsesDevice.
 */
void AllSpikingSynapses::setAdvanceSynapsesDeviceParams()
{
    setSynapseClassID();
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
void AllSpikingSynapses::setSynapseClassID()
{
    enumClassSynapses classSynapses_h = classAllSpikingSynapses;

    HANDLE_ERROR( cudaMemcpyToSymbol( classSynapses_d, &classSynapses_h, sizeof(enumClassSynapses) ) );
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
void AllSpikingSynapses::advanceSynapses(void* allSynapsesDevice, void* allNeuronsDevice, void* synapseIndexMapDevice )
{
    if (totalSynapseCount_ == 0)
        return;

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( totalSynapseCount_ + threadsPerBlock - 1 ) / threadsPerBlock;

    // Advance synapses ------------->
    advanceSpikingSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( totalSynapseCount_, (SynapseIndexMap*) synapseIndexMapDevice, g_simulationStep, Simulator::getInstance().getDeltaT(), (AllSpikingSynapsesDeviceProperties*)allSynapsesDevice );
}

/*
 *  Prints GPU SynapsesProps data.
 *   
 *  @param  allSynapsesDeviceProps   GPU address of the corresponding SynapsesDeviceProperties struct on device memory.
 */
void AllSpikingSynapses::printGPUSynapsesProps( void* allSynapsesDeviceProps ) const
{
    AllSpikingSynapsesDeviceProperties allSynapsesProps;

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


        // copy everything
        HANDLE_ERROR( cudaMemcpy ( &allSynapsesProps, allSynapsesDeviceProps, sizeof( AllSpikingSynapsesDeviceProperties), cudaMemcpyDeviceToHost ) );
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
                cout << " GPU total_delay: " << totalDelayPrint[i] << endl;;
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
    }
}


