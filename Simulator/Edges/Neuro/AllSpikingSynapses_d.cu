/**
 * @file AllSpikingSynapses_d.cu
 * 
 * @ingroup Simulator/Edges
 *
 * @brief
 */

#include "AllSpikingSynapses.h"
#include "AllSynapsesDeviceFuncs.h"
#include "Book.h"

///  Allocate GPU memories to store all synapses' states,
///  and copy them from host to GPU memory.
///
///  @param  allSynapsesDevice  GPU address of the AllSpikingSynapsesDeviceProperties struct 
///                             on device memory.
void AllSpikingSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice ) {
        allocSynapseDeviceStruct( allSynapsesDevice, Simulator::getInstance().getTotalVertices(), Simulator::getInstance().getMaxSynapsesPerNeuron() );
}

///  Allocate GPU memories to store all synapses' states,
///  and copy them from host to GPU memory.
///
///  @param  allSynapsesDevice     GPU address of the AllSpikingSynapsesDeviceProperties struct 
///                                on device memory.
///  @param  numNeurons            Number of neurons.
///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
void AllSpikingSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, int numNeurons, int maxEdgesPerVertex ) {
        AllSpikingSynapsesDeviceProperties allSynapses;

        allocDeviceStruct( allSynapses, numNeurons, maxEdgesPerVertex );

        HANDLE_ERROR( cudaMalloc( allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ) ) );
        HANDLE_ERROR( cudaMemcpy ( *allSynapsesDevice, &allSynapses, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyHostToDevice ) );
}

///  Allocate GPU memories to store all synapses' states,
///  and copy them from host to GPU memory.
///  (Helper function of allocSynapseDeviceStruct)
///
///  @param  allSynapsesDevice     GPU address of the AllSpikingSynapsesDeviceProperties struct 
///                                on device memory.
///  @param  numNeurons            Number of neurons.
///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
void AllSpikingSynapses::allocDeviceStruct( AllSpikingSynapsesDeviceProperties &allSynapsesDevice, int numNeurons, int maxEdgesPerVertex ) {
        BGSIZE maxTotalSynapses = maxEdgesPerVertex * numNeurons;

        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.sourceNeuronIndex_, maxTotalSynapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.destNeuronIndex_, maxTotalSynapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.W_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.type_, maxTotalSynapses * sizeof( synapseType ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.psr_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.inUse_, maxTotalSynapses * sizeof( bool ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.synapseCounts_, numNeurons * sizeof( BGSIZE ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.decay_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.tau_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.totalDelay_, maxTotalSynapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.delayQueue_, maxTotalSynapses * sizeof( uint32_t ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.delayIndex_, maxTotalSynapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice.delayQueueLength_, maxTotalSynapses * sizeof( int ) ) );
}

///  Delete GPU memories.
///
///  @param  allSynapsesDevice  GPU address of the AllSpikingSynapsesDeviceProperties struct 
///                             on device memory.
void AllSpikingSynapses::deleteSynapseDeviceStruct( void* allSynapsesDevice ) {
        AllSpikingSynapsesDeviceProperties allSynapses;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

        deleteDeviceStruct( allSynapses );

        HANDLE_ERROR( cudaFree( allSynapsesDevice ) );
}

///  Delete GPU memories.
///  (Helper function of deleteSynapseDeviceStruct)
///
///  @param  allSynapsesDevice  GPU address of the AllSpikingSynapsesDeviceProperties struct 
///                             on device memory.
void AllSpikingSynapses::deleteDeviceStruct( AllSpikingSynapsesDeviceProperties& allSynapsesDevice ) {
        HANDLE_ERROR( cudaFree( allSynapsesDevice.sourceNeuronIndex_ ) );
        HANDLE_ERROR( cudaFree( allSynapsesDevice.destNeuronIndex_ ) );
        HANDLE_ERROR( cudaFree( allSynapsesDevice.W_ ) );
        HANDLE_ERROR( cudaFree( allSynapsesDevice.type_ ) );
        HANDLE_ERROR( cudaFree( allSynapsesDevice.psr_ ) );
        HANDLE_ERROR( cudaFree( allSynapsesDevice.inUse_ ) );
        HANDLE_ERROR( cudaFree( allSynapsesDevice.synapseCounts_ ) );
        HANDLE_ERROR( cudaFree( allSynapsesDevice.decay_ ) );
        HANDLE_ERROR( cudaFree( allSynapsesDevice.tau_ ) );
        HANDLE_ERROR( cudaFree( allSynapsesDevice.totalDelay_ ) );
        HANDLE_ERROR( cudaFree( allSynapsesDevice.delayQueue_ ) );
        HANDLE_ERROR( cudaFree( allSynapsesDevice.delayIndex_ ) );
        HANDLE_ERROR( cudaFree( allSynapsesDevice.delayQueueLength_ ) );

        // Set countVertices_ to 0 to avoid illegal memory deallocation 
        // at AllSpikingSynapses deconstructor.
        //allSynapses.countVertices_ = 0;
}

///  Copy all synapses' data from host to device.
///
///  @param  allSynapsesDevice  GPU address of the AllSpikingSynapsesDeviceProperties struct 
///                             on device memory.
void AllSpikingSynapses::copySynapseHostToDevice( void* allSynapsesDevice ) { // copy everything necessary
        copySynapseHostToDevice( allSynapsesDevice, Simulator::getInstance().getTotalVertices(), Simulator::getInstance().getMaxSynapsesPerNeuron() );
}

///  Copy all synapses' data from host to device.
///
///  @param  allSynapsesDevice     GPU address of the AllSpikingSynapsesDeviceProperties struct 
///                                on device memory.
///  @param  numNeurons            Number of neurons.
///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
void AllSpikingSynapses::copySynapseHostToDevice( void* allSynapsesDevice, int numNeurons, int maxEdgesPerVertex ) { // copy everything necessary
        AllSpikingSynapsesDeviceProperties allSynapsesDeviceProps;

        HANDLE_ERROR( cudaMemcpy ( &allSynapsesDeviceProps, allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

        copyHostToDevice( allSynapsesDevice, allSynapsesDeviceProps, numNeurons, maxEdgesPerVertex );
}

///  Copy all synapses' data from host to device.
///  (Helper function of copySynapseHostToDevice)
///
///  @param  allSynapsesDevice           GPU address of the allSynapses struct on device memory.     
///  @param  allSynapsesDeviceProps      GPU address of the AllSpikingSynapsesDeviceProperties struct on device memory.
///  @param  numNeurons                  Number of neurons.
///  @param  maxEdgesPerVertex        Maximum number of synapses per neuron.
void AllSpikingSynapses::copyHostToDevice( void* allSynapsesDevice, AllSpikingSynapsesDeviceProperties& allSynapsesDeviceProps, int numNeurons, int maxEdgesPerVertex ) { // copy everything necessary 
        BGSIZE maxTotalSynapses = maxEdgesPerVertex * numNeurons;

        allSynapsesDeviceProps.maxEdgesPerVertex_ = maxEdgesPerVertex_;
        allSynapsesDeviceProps.totalEdgeCount_ = totalEdgeCount_;
        allSynapsesDeviceProps.countVertices_ = countVertices_;
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDevice, &allSynapsesDeviceProps, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyHostToDevice ) );

        // Set countVertices_ to 0 to avoid illegal memory deallocation 
        // at AllSpikingSynapses deconstructor.
        allSynapsesDeviceProps.countVertices_ = 0;

        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.sourceNeuronIndex_, sourceNeuronIndex_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.destNeuronIndex_, destNeuronIndex_,
                maxTotalSynapses * sizeof( int ),  cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.W_, W_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.type_, type_,
                maxTotalSynapses * sizeof( synapseType ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.psr_, psr_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.inUse_, inUse_,
                maxTotalSynapses * sizeof( bool ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.synapseCounts_, synapseCounts_,
                        numNeurons * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.decay_, decay_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.tau_, tau_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.totalDelay_, totalDelay_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.delayQueue_, delayQueue_,
                maxTotalSynapses * sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.delayIndex_, delayIndex_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.delayQueueLength_, delayQueueLength_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyHostToDevice ) );
}

///  Copy all synapses' data from device to host.
///
///  @param  allSynapsesDevice  GPU address of the AllSpikingSynapsesDeviceProperties struct 
///                             on device memory.
void AllSpikingSynapses::copySynapseDeviceToHost( void* allSynapsesDevice ) {
        // copy everything necessary
        AllSpikingSynapsesDeviceProperties allSynapses;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

        copyDeviceToHost( allSynapses );
}

///  Copy all synapses' data from device to host.
///  (Helper function of copySynapseDeviceToHost)
///
///  @param  allSynapsesDevice     GPU address of the AllSpikingSynapsesDeviceProperties struct 
///                                on device memory.
///  @param  numNeurons            Number of neurons.
///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
void AllSpikingSynapses::copyDeviceToHost( AllSpikingSynapsesDeviceProperties& allSynapsesDevice ) {
        int numNeurons = Simulator::getInstance().getTotalVertices();
        BGSIZE maxTotalSynapses = Simulator::getInstance().getMaxSynapsesPerNeuron() * numNeurons;

        HANDLE_ERROR( cudaMemcpy ( synapseCounts_, allSynapsesDevice.synapseCounts_,
                numNeurons * sizeof( BGSIZE ), cudaMemcpyDeviceToHost ) );
        maxEdgesPerVertex_ = allSynapsesDevice.maxEdgesPerVertex_;
        totalEdgeCount_ = allSynapsesDevice.totalEdgeCount_;
        countVertices_ = allSynapsesDevice.countVertices_;

        // Set countVertices_ to 0 to avoid illegal memory deallocation 
        // at AllSpikingSynapses deconstructor.
        allSynapsesDevice.countVertices_ = 0;

        HANDLE_ERROR( cudaMemcpy ( sourceNeuronIndex_, allSynapsesDevice.sourceNeuronIndex_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( destNeuronIndex_, allSynapsesDevice.destNeuronIndex_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( W_, allSynapsesDevice.W_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( type_, allSynapsesDevice.type_,
                maxTotalSynapses * sizeof( synapseType ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( psr_, allSynapsesDevice.psr_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( inUse_, allSynapsesDevice.inUse_,
                maxTotalSynapses * sizeof( bool ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( decay_, allSynapsesDevice.decay_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tau_, allSynapsesDevice.tau_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( totalDelay_, allSynapsesDevice.totalDelay_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( delayQueue_, allSynapsesDevice.delayQueue_,
                maxTotalSynapses * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( delayIndex_, allSynapsesDevice.delayIndex_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( delayQueueLength_, allSynapsesDevice.delayQueueLength_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
}

///  Get synapse_counts in AllEdges struct on device memory.
///
///  @param  allSynapsesDevice  GPU address of the AllSpikingSynapsesDeviceProperties struct 
///                             on device memory.
void AllSpikingSynapses::copyDeviceSynapseCountsToHost( void* allSynapsesDevice )
{
        AllSpikingSynapsesDeviceProperties allSynapsesDeviceProps;
        int neuronCount = Simulator::getInstance().getTotalVertices();

        HANDLE_ERROR( cudaMemcpy ( &allSynapsesDeviceProps, allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( synapseCounts_, allSynapsesDeviceProps.synapseCounts_, neuronCount * sizeof( BGSIZE ), cudaMemcpyDeviceToHost ) );

        // Set countVertices_ to 0 to avoid illegal memory deallocation 
        // at AllSpikingSynapses deconstructor.
        //allSynapses.countVertices_ = 0;
}

///  Get summationCoord and in_use in AllEdges struct on device memory.
///
///  @param  allSynapsesDevice  GPU address of the AllSpikingSynapsesDeviceProperties struct
///                             on device memory.
void AllSpikingSynapses::copyDeviceSynapseSumIdxToHost(void* allSynapsesDevice )
{
        AllSpikingSynapsesDeviceProperties allSynapsesDeviceProps;
        BGSIZE maxTotalSynapses = Simulator::getInstance().getMaxSynapsesPerNeuron() * Simulator::getInstance().getTotalVertices();

        HANDLE_ERROR( cudaMemcpy ( &allSynapsesDeviceProps, allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( sourceNeuronIndex_, allSynapsesDeviceProps.sourceNeuronIndex_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( inUse_, allSynapsesDeviceProps.inUse_,
                maxTotalSynapses * sizeof( bool ), cudaMemcpyDeviceToHost ) );
       
        // Set countVertices_ to 0 to avoid illegal memory deallocation 
        // at AllSpikingSynapses deconstructor.
        //allSynapses.countVertices_ = 0;
}

///  Set some parameters used for advanceSynapsesDevice.
void AllSpikingSynapses::setAdvanceSynapsesDeviceParams()
{
    setSynapseClassID();
}

///  Set synapse class ID defined by enumClassSynapses for the caller's Synapse class.
///  The class ID will be set to classSynapses_d in device memory,
///  and the classSynapses_d will be referred to call a device function for the
///  particular synapse class.
///  Because we cannot use virtual function (Polymorphism) in device functions,
///  we use this scheme.
///  Note: we used to use a function pointer; however, it caused the growth_cuda crash
///  (see issue#137).
void AllSpikingSynapses::setSynapseClassID()
{
    enumClassSynapses classSynapses_h = classAllSpikingSynapses;

    HANDLE_ERROR( cudaMemcpyToSymbol( classSynapses_d, &classSynapses_h, sizeof(enumClassSynapses) ) );
}

///  Advance all the Synapses in the simulation.
///  Update the state of all synapses for a time step.
///
///  @param  allSynapsesDevice      GPU address of the AllSynapsesDeviceProperties struct
///                                 on device memory.
///  @param  allNeuronsDevice       GPU address of the allNeurons struct on device memory.
///  @param  synapseIndexMapDevice  GPU address of the EdgeIndexMap on device memory.
void AllSpikingSynapses::advanceEdges(void* allSynapsesDevice, void* allNeuronsDevice, void* synapseIndexMapDevice )
{
    if (totalEdgeCount_ == 0)
        return;

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( totalEdgeCount_ + threadsPerBlock - 1 ) / threadsPerBlock;

    // Advance synapses ------------->
    advanceSpikingSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( totalEdgeCount_, (EdgeIndexMap*) synapseIndexMapDevice, g_simulationStep, Simulator::getInstance().getDeltaT(), (AllSpikingSynapsesDeviceProperties*)allSynapsesDevice );
}

///  Prints GPU SynapsesProps data.
///   
///  @param  allSynapsesDeviceProps   GPU address of the corresponding SynapsesDeviceProperties struct on device memory.
void AllSpikingSynapses::printGPUSynapsesProps( void* allSynapsesDeviceProps ) const
{
    AllSpikingSynapsesDeviceProperties allSynapsesProps;

    //allocate print out data members
    BGSIZE size = maxEdgesPerVertex_ * countVertices_;
    if (size != 0) {
        BGSIZE *synapseCountsPrint = new BGSIZE[countVertices_];
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

        for (int i = 0; i < countVertices_; i++) {
            synapseCountsPrint[i] = 0;
        }

        BGFLOAT *decayPrint = new BGFLOAT[size];
        int *totalDelayPrint = new int[size];
        BGFLOAT *tauPrint = new BGFLOAT[size];


        // copy everything
        HANDLE_ERROR( cudaMemcpy ( &allSynapsesProps, allSynapsesDeviceProps, sizeof( AllSpikingSynapsesDeviceProperties), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( synapseCountsPrint, allSynapsesProps.synapseCounts_, countVertices_ * sizeof( BGSIZE ), cudaMemcpyDeviceToHost ) );
        maxSynapsesPerNeuronPrint = allSynapsesProps.maxEdgesPerVertex_;
        totalSynapseCountPrint = allSynapsesProps.totalEdgeCount_;
        countNeuronsPrint = allSynapsesProps.countVertices_;

        // Set countVertices_ to 0 to avoid illegal memory deallocation
        // at AllSynapsesProps deconstructor.
        allSynapsesProps.countVertices_ = 0;

        HANDLE_ERROR( cudaMemcpy ( sourceNeuronIndexPrint, allSynapsesProps.sourceNeuronIndex_, size * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( destNeuronIndexPrint, allSynapsesProps.destNeuronIndex_, size * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( WPrint, allSynapsesProps.W_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( typePrint, allSynapsesProps.type_, size * sizeof( synapseType ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( psrPrint, allSynapsesProps.psr_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( inUsePrint, allSynapsesProps.inUse_, size * sizeof( bool ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( decayPrint, allSynapsesProps.decay_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tauPrint, allSynapsesProps.tau_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( totalDelayPrint, allSynapsesProps.totalDelay_, size * sizeof( int ), cudaMemcpyDeviceToHost ) );


        for(int i = 0; i < maxEdgesPerVertex_ * countVertices_; i++) {
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

        for (int i = 0; i < countVertices_; i++) {
            cout << "GPU synapse_counts:" << "neuron[" << i  << "]" << synapseCountsPrint[i] << endl;
        }

        cout << "GPU totalSynapseCount:" << totalSynapseCountPrint << endl;
        cout << "GPU maxEdgesPerVertex:" << maxSynapsesPerNeuronPrint << endl;
        cout << "GPU countVertices_:" << countNeuronsPrint << endl;


        // Set countVertices_ to 0 to avoid illegal memory deallocation
        // at AllDSSynapsesProps deconstructor.
        allSynapsesProps.countVertices_ = 0;

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


