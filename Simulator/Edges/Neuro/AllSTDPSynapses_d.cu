/**
 * @file AllSTDPSynapses_d.cu
 * 
 * @ingroup Simulator/Edges
 *
 * @brief A container of all STDP synapse data
 */

#include "AllSTDPSynapses.h"
#include "AllSpikingSynapses.h"
#include "GPUModel.h"
#include "AllSynapsesDeviceFuncs.h"
#include "Book.h"

///  Allocate GPU memories to store all synapses' states,
///  and copy them from host to GPU memory.
///
///  @param  allEdgesDevice  GPU address of the AllSTDPSynapsesDeviceProperties struct 
///                             on device memory.
void AllSTDPSynapses::allocEdgeDeviceStruct( void** allEdgesDevice ) {
	allocEdgeDeviceStruct( allEdgesDevice, Simulator::getInstance().getTotalVertices(), Simulator::getInstance().getMaxEdgesPerVertex() );
}

///  Allocate GPU memories to store all synapses' states,
///  and copy them from host to GPU memory.
///
///  @param  allEdgesDevice     GPU address of the AllSTDPSynapsesDeviceProperties struct 
///                                on device memory.
///  @param  numVertices            Number of vertices.
///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
void AllSTDPSynapses::allocEdgeDeviceStruct( void** allEdgesDevice, int numVertices, int maxEdgesPerVertex ) {
	AllSTDPSynapsesDeviceProperties allEdgesDeviceProps;

	allocDeviceStruct( allEdgesDeviceProps, numVertices, maxEdgesPerVertex );

	HANDLE_ERROR( cudaMalloc( allEdgesDevice, sizeof( AllSTDPSynapsesDeviceProperties ) ) );
	HANDLE_ERROR( cudaMemcpy ( *allEdgesDevice, &allEdgesDeviceProps, sizeof( AllSTDPSynapsesDeviceProperties ), cudaMemcpyHostToDevice ) );
}

///  Allocate GPU memories to store all synapses' states,
///  and copy them from host to GPU memory.
///  (Helper function of allocEdgeDeviceStruct)
///
///  @param  allEdgesDevice     GPU address of the AllSTDPSynapsesDeviceProperties struct 
///                                on device memory.
///  @param  numVertices            Number of vertices.
///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
void AllSTDPSynapses::allocDeviceStruct( AllSTDPSynapsesDeviceProperties &allEdgesDevice, int numVertices, int maxEdgesPerVertex ) {
        AllSpikingSynapses::allocDeviceStruct( allEdgesDevice, numVertices, maxEdgesPerVertex );

        BGSIZE maxTotalSynapses = maxEdgesPerVertex * numVertices;

        HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdgesDevice.totalDelayPost_, maxTotalSynapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdgesDevice.delayQueuePost_, maxTotalSynapses * sizeof( BGSIZE ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdgesDevice.delayIndexPost_, maxTotalSynapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdgesDevice.delayQueuePost_, maxTotalSynapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdgesDevice.tauspost_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdgesDevice.tauspre_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdgesDevice.taupos_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdgesDevice.tauneg_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdgesDevice.STDPgap_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdgesDevice.Wex_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdgesDevice.Aneg_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdgesDevice.Apos_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdgesDevice.mupos_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdgesDevice.muneg_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
}

///  Delete GPU memories.
///
///  @param  allEdgesDevice  GPU address of the AllSTDPSynapsesDeviceProperties struct 
///                             on device memory.
void AllSTDPSynapses::deleteEdgeDeviceStruct( void* allEdgesDevice ) {
	AllSTDPSynapsesDeviceProperties allEdgesDeviceProps;

	HANDLE_ERROR( cudaMemcpy ( &allEdgesDeviceProps, allEdgesDevice, sizeof( AllSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allEdgesDeviceProps );

	HANDLE_ERROR( cudaFree( allEdgesDevice ) );
}

///  Delete GPU memories.
///  (Helper function of deleteEdgeDeviceStruct)
///
///  @param  allEdgesDevice  GPU address of the AllSTDPSynapsesDeviceProperties struct 
///                             on device memory.
void AllSTDPSynapses::deleteDeviceStruct( AllSTDPSynapsesDeviceProperties& allEdgesDevice ) {
        HANDLE_ERROR( cudaFree( allEdgesDevice.totalDelayPost_ ) );
        HANDLE_ERROR( cudaFree( allEdgesDevice.delayQueuePost_ ) );
        HANDLE_ERROR( cudaFree( allEdgesDevice.delayIndexPost_ ) );
        HANDLE_ERROR( cudaFree( allEdgesDevice.tauspost_ ) );
        HANDLE_ERROR( cudaFree( allEdgesDevice.tauspre_ ) );
        HANDLE_ERROR( cudaFree( allEdgesDevice.taupos_ ) );
        HANDLE_ERROR( cudaFree( allEdgesDevice.tauneg_ ) );
        HANDLE_ERROR( cudaFree( allEdgesDevice.STDPgap_ ) );
        HANDLE_ERROR( cudaFree( allEdgesDevice.Wex_ ) );
        HANDLE_ERROR( cudaFree( allEdgesDevice.Aneg_ ) );
        HANDLE_ERROR( cudaFree( allEdgesDevice.Apos_ ) );
        HANDLE_ERROR( cudaFree( allEdgesDevice.mupos_ ) );
        HANDLE_ERROR( cudaFree( allEdgesDevice.muneg_) );

        AllSpikingSynapses::deleteDeviceStruct( allEdgesDevice );
}

///  Copy all synapses' data from host to device.
///
///  @param  allEdgesDevice     GPU address of the AllSTDPSynapsesDeviceProperties struct 
///                                on device memory.
///  @param  numVertices            Number of vertices.
///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
void AllSTDPSynapses::copyEdgeHostToDevice( void* allEdgesDevice ) { // copy everything necessary
	copyEdgeHostToDevice( allEdgesDevice, Simulator::getInstance().getTotalVertices(), Simulator::getInstance().getMaxEdgesPerVertex() );	
}

///  Copy all synapses' data from host to device.
///
///  @param  allEdgesDevice     GPU address of the AllSTDPSynapsesDeviceProperties struct 
///                                on device memory.
///  @param  numVertices            Number of vertices.
///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
void AllSTDPSynapses::copyEdgeHostToDevice( void* allEdgesDevice, int numVertices, int maxEdgesPerVertex ) { // copy everything necessary
	AllSTDPSynapsesDeviceProperties allEdgesDeviceProps;

        HANDLE_ERROR( cudaMemcpy ( &allEdgesDeviceProps, allEdgesDevice, sizeof( AllSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	copyHostToDevice( allEdgesDevice, allEdgesDeviceProps, numVertices, maxEdgesPerVertex );	
}

///  Copy all synapses' data from host to device.
///  (Helper function of copyEdgeHostToDevice)
///
///  @param  allEdgesDevice     GPU address of the AllSTDPSynapsesDeviceProperties struct 
///                                on device memory.
///  @param  numVertices            Number of vertices.
///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
void AllSTDPSynapses::copyHostToDevice( void* allEdgesDevice, AllSTDPSynapsesDeviceProperties& allEdgesDeviceProps, int numVertices, int maxEdgesPerVertex ) { // copy everything necessary 
        AllSpikingSynapses::copyHostToDevice( allEdgesDevice, allEdgesDeviceProps, numVertices, maxEdgesPerVertex );

        BGSIZE maxTotalSynapses = maxEdgesPerVertex * numVertices;
        
        HANDLE_ERROR( cudaMemcpy ( allEdgesDeviceProps.totalDelayPost_, totalDelayPost_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allEdgesDeviceProps.delayQueuePost_, delayQueuePost_,
                maxTotalSynapses * sizeof( uint32_t ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allEdgesDeviceProps.delayIndexPost_, delayIndexPost_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allEdgesDeviceProps.delayQueuePost_, delayQueuePost_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allEdgesDeviceProps.tauspost_, tauspost_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allEdgesDeviceProps.tauspre_, tauspre_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allEdgesDeviceProps.taupos_, taupos_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allEdgesDeviceProps.tauneg_, tauneg_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allEdgesDeviceProps.STDPgap_, STDPgap_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allEdgesDeviceProps.Wex_, Wex_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allEdgesDeviceProps.Aneg_, Aneg_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allEdgesDeviceProps.Apos_, Apos_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allEdgesDeviceProps.mupos_, mupos_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allEdgesDeviceProps.muneg_, muneg_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
}

///  Copy all synapses' data from device to host.
///
///  @param  allEdgesDevice  GPU address of the AllSTDPSynapsesDeviceProperties struct 
///                             on device memory.
void AllSTDPSynapses::copyEdgeDeviceToHost( void* allEdgesDevice ) {
	// copy everything necessary
	AllSTDPSynapsesDeviceProperties allEdgesDeviceProps;

        HANDLE_ERROR( cudaMemcpy ( &allEdgesDeviceProps, allEdgesDevice, sizeof( AllSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	copyDeviceToHost( allEdgesDeviceProps );
}

///  Copy all synapses' data from device to host.
///  (Helper function of copyEdgeDeviceToHost)
///
///  @param  allEdgesDevice     GPU address of the AllSTDPSynapsesDeviceProperties struct 
///                                on device memory.
///  @param  numVertices            Number of vertices.
///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
void AllSTDPSynapses::copyDeviceToHost( AllSTDPSynapsesDeviceProperties& allEdgesDevice ) {
        AllSpikingSynapses::copyDeviceToHost( allEdgesDevice ) ;

	int numVertices = Simulator::getInstance().getTotalVertices();
	BGSIZE maxTotalSynapses = Simulator::getInstance().getMaxEdgesPerVertex() * numVertices;

        HANDLE_ERROR( cudaMemcpy ( delayQueuePost_, allEdgesDevice.delayQueuePost_,
                maxTotalSynapses * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( delayIndexPost_, allEdgesDevice.delayIndexPost_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( delayQueuePost_, allEdgesDevice.delayQueuePost_,
                maxTotalSynapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tauspost_, allEdgesDevice.tauspost_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tauspre_, allEdgesDevice.tauspre_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( taupos_, allEdgesDevice.taupos_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tauneg_, allEdgesDevice.tauneg_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( STDPgap_, allEdgesDevice.STDPgap_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( Wex_, allEdgesDevice.Wex_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( Aneg_, allEdgesDevice.Aneg_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( Apos_, allEdgesDevice.Apos_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( mupos_, allEdgesDevice.mupos_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( muneg_, allEdgesDevice.muneg_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
}

///  Advance all the Synapses in the simulation.
///  Update the state of all synapses for a time step.
///
///  @param  allEdgesDevice      GPU address of the AllEdgesDeviceProperties struct 
///                                 on device memory.
///  @param  allVerticesDevice       GPU address of the allNeurons struct on device memory.
///  @param  edgeIndexMapDevice  GPU address of the EdgeIndexMap on device memory.
void AllSTDPSynapses::advanceEdges( void* allEdgesDevice, void* allVerticesDevice, void* edgeIndexMapDevice )
{
    int maxSpikes = (int) ((Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getMaxFiringRate()));

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( totalEdgeCount_ + threadsPerBlock - 1 ) / threadsPerBlock;
    // Advance synapses ------------->
    advanceSTDPSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( totalEdgeCount_, (EdgeIndexMap*) edgeIndexMapDevice, g_simulationStep, Simulator::getInstance().getDeltaT(), 
                                (AllSTDPSynapsesDeviceProperties*)allEdgesDevice, (AllSpikingNeuronsDeviceProperties*)allVerticesDevice, maxSpikes );
}
    
///  Set synapse class ID defined by enumClassSynapses for the caller's Synapse class.
///  The class ID will be set to classSynapses_d in device memory,
///  and the classSynapses_d will be referred to call a device function for the
///  particular synapse class.
///  Because we cannot use virtual function (Polymorphism) in device functions,
///  we use this scheme.
///  Note: we used to use a function pointer; however, it caused the growth_cuda crash
///  (see issue#137).
void AllSTDPSynapses::setEdgeClassID()
{
    enumClassSynapses classSynapses_h = classAllSTDPSynapses;

    HANDLE_ERROR( cudaMemcpyToSymbol(classSynapses_d, &classSynapses_h, sizeof(enumClassSynapses)) );
}

///  Prints GPU SynapsesProps data.
///   
///  @param  allEdgesDeviceProps   GPU address of the corresponding SynapsesDeviceProperties struct on device memory.
void AllSTDPSynapses::printGPUEdgesProps( void* allEdgesDeviceProps ) const
{
    AllSTDPSynapsesDeviceProperties allSynapsesProps;

    //allocate print out data members
    BGSIZE size = maxEdgesPerVertex_ * countVertices_;
    if (size != 0) {
        BGSIZE *synapseCountsPrint = new BGSIZE[countVertices_];
        BGSIZE maxEdgesPerVertexPrint;
        BGSIZE totalSynapseCountPrint;
        int countNeuronsPrint;
        int *sourceNeuronIndexPrint = new int[size];
        int *destNeuronIndexPrint = new int[size];
        BGFLOAT *WPrint = new BGFLOAT[size];

        edgeType *typePrint = new edgeType[size];
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

        // copy everything
        HANDLE_ERROR( cudaMemcpy ( &allSynapsesProps, allEdgesDeviceProps, sizeof( AllSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( synapseCountsPrint, allSynapsesProps.edgeCounts_, countVertices_ * sizeof( BGSIZE ), cudaMemcpyDeviceToHost ) );
        maxEdgesPerVertexPrint = allSynapsesProps.maxEdgesPerVertex_;
        totalSynapseCountPrint = allSynapsesProps.totalEdgeCount_;
        countNeuronsPrint = allSynapsesProps.countVertices_;

        // Set countVertices_ to 0 to avoid illegal memory deallocation
        // at AllSynapsesProps deconstructor.
        allSynapsesProps.countVertices_ = 0;

        HANDLE_ERROR( cudaMemcpy ( sourceNeuronIndexPrint, allSynapsesProps.sourceVertexIndex_, size * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( destNeuronIndexPrint, allSynapsesProps.destVertexIndex_, size * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( WPrint, allSynapsesProps.W_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( typePrint, allSynapsesProps.type_, size * sizeof( edgeType ), cudaMemcpyDeviceToHost ) );
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
                cout << " GPU muneg_: " << munegPrint[i] << endl;
            }
        }

        for (int i = 0; i < countVertices_; i++) {
            cout << "GPU edge_counts:" << "neuron[" << i  << "]" << synapseCountsPrint[i] << endl;
        }

        cout << "GPU totalSynapseCount:" << totalSynapseCountPrint << endl;
        cout << "GPU maxEdgesPerVertex:" << maxEdgesPerVertexPrint << endl;
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
        destNeuronIndexPrint = nullptr;
        WPrint = nullptr;
        sourceNeuronIndexPrint = nullptr;
        psrPrint = nullptr;
        typePrint = nullptr;
        inUsePrint = nullptr;
        synapseCountsPrint = nullptr;

        delete[] decayPrint;
        delete[] totalDelayPrint;
        delete[] tauPrint;
        decayPrint = nullptr;
        totalDelayPrint = nullptr;
        tauPrint = nullptr;

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
        totalDelayPostPrint = nullptr;
        tauspostPrint = nullptr;
        tausprePrint = nullptr;
        tauposPrint = nullptr;
        taunegPrint = nullptr;
        STDPgapPrint = nullptr;
        WexPrint = nullptr;
        AnegPrint = nullptr;
        AposPrint = nullptr;
        muposPrint = nullptr;
        munegPrint = nullptr;
    }

}
