/**
 * @file AllDynamicSTDPSynapses_d.cu
 *
 * @ingroup Simulator/Edges
 * 
 * @brief A container of all dynamic STDP synapse data
 */

#include "AllDynamicSTDPSynapses.h"
#include "AllSynapsesDeviceFuncs.h"
#include "Book.h"
#include "Simulator.h"

///  Allocate GPU memories to store all synapses' states,
///  and copy them from host to GPU memory.
///
///  @param  allEdgesDevice  GPU address of the AllDynamicSTDPSynapsesDeviceProperties struct 
///                             on device memory.
void AllDynamicSTDPSynapses::allocEdgeDeviceStruct( void** allEdgesDevice ) {
	allocEdgeDeviceStruct( allEdgesDevice, Simulator::getInstance().getTotalVertices(), Simulator::getInstance().getMaxEdgesPerVertex() );
}

///  Allocate GPU memories to store all synapses' states,
///  and copy them from host to GPU memory.
///
///  @param  allEdgesDevice     GPU address of the AllDynamicSTDPSynapsesDeviceProperties struct 
///                                on device memory.
///  @param  numVertices            Number of vertices.
///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
void AllDynamicSTDPSynapses::allocEdgeDeviceStruct( void** allEdgesDevice, int numVertices, int maxEdgesPerVertex ) {
	AllDynamicSTDPSynapsesDeviceProperties allEdges;

	allocDeviceStruct( allEdges, numVertices, maxEdgesPerVertex );

	HANDLE_ERROR( cudaMalloc( allEdgesDevice, sizeof( AllDynamicSTDPSynapsesDeviceProperties ) ) );
	HANDLE_ERROR( cudaMemcpy ( *allEdgesDevice, &allEdges, sizeof( AllDynamicSTDPSynapsesDeviceProperties ), cudaMemcpyHostToDevice ) );
}

///  Allocate GPU memories to store all synapses' states,
///  and copy them from host to GPU memory.
///  (Helper function of allocEdgeDeviceStruct)
///
///  @param  allEdgesDeviceProps      GPU address of the AllDynamicSTDPSynapsesDeviceProperties struct 
///                                      on device memory.
///  @param  numVertices                  Number of vertices.
///  @param  maxEdgesPerVertex        Maximum number of synapses per neuron.
void AllDynamicSTDPSynapses::allocDeviceStruct( AllDynamicSTDPSynapsesDeviceProperties &allEdgesDeviceProps, int numVertices, int maxEdgesPerVertex ) {
        AllSTDPSynapses::allocDeviceStruct( allEdgesDeviceProps, numVertices, maxEdgesPerVertex );

        BGSIZE maxTotalSynapses = maxEdgesPerVertex * numVertices;

        HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdgesDeviceProps.lastSpike_, maxTotalSynapses * sizeof( uint64_t ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdgesDeviceProps.r_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdgesDeviceProps.u_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdgesDeviceProps.D_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdgesDeviceProps.U_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdgesDeviceProps.F_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
}

///  Delete GPU memories.
///
///  @param  allEdgesDevice  GPU address of the AllDynamicSTDPSynapsesDeviceProperties struct 
///                             on device memory.
void AllDynamicSTDPSynapses::deleteEdgeDeviceStruct( void* allEdgesDevice ) {
	AllDynamicSTDPSynapsesDeviceProperties allEdges;

	HANDLE_ERROR( cudaMemcpy ( &allEdges, allEdgesDevice, sizeof( AllDynamicSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allEdges );

	HANDLE_ERROR( cudaFree( allEdgesDevice ) );
}

///  Delete GPU memories.
///  (Helper function of deleteEdgeDeviceStruct)
///
///  @param  allEdgesDeviceProps  GPU address of the AllDynamicSTDPSynapsesDeviceProperties struct 
///                                  on device memory.
void AllDynamicSTDPSynapses::deleteDeviceStruct( AllDynamicSTDPSynapsesDeviceProperties& allEdgesDeviceProps ) {
        HANDLE_ERROR( cudaFree( allEdgesDeviceProps.lastSpike_ ) );
	HANDLE_ERROR( cudaFree( allEdgesDeviceProps.r_ ) );
	HANDLE_ERROR( cudaFree( allEdgesDeviceProps.u_ ) );
	HANDLE_ERROR( cudaFree( allEdgesDeviceProps.D_ ) );
	HANDLE_ERROR( cudaFree( allEdgesDeviceProps.U_ ) );
	HANDLE_ERROR( cudaFree( allEdgesDeviceProps.F_ ) );

        AllSTDPSynapses::deleteDeviceStruct( allEdgesDeviceProps );
}

///  Copy all synapses' data from host to device.
///
///  @param  allEdgesDevice  GPU address of the AllDynamicSTDPSynapsesDeviceProperties struct 
///                             on device memory.
void AllDynamicSTDPSynapses::copyEdgeHostToDevice( void* allEdgesDevice ) { // copy everything necessary
	copyEdgeHostToDevice( allEdgesDevice, Simulator::getInstance().getTotalVertices(), Simulator::getInstance().getMaxEdgesPerVertex() );	
}

///  Copy all synapses' data from host to device.
///
///  @param  allEdgesDevice     GPU address of the AllDynamicSTDPSynapsesDeviceProperties struct 
///                                on device memory.
///  @param  numVertices            Number of vertices.
///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
void AllDynamicSTDPSynapses::copyEdgeHostToDevice( void* allEdgesDevice, int numVertices, int maxEdgesPerVertex ) { // copy everything necessary
	AllDynamicSTDPSynapsesDeviceProperties allEdges;

        HANDLE_ERROR( cudaMemcpy ( &allEdges, allEdgesDevice, sizeof( AllDynamicSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	copyHostToDevice( allEdgesDevice, allEdges, numVertices, maxEdgesPerVertex );	
}

///  Copy all synapses' data from host to device.
///  (Helper function of copyEdgeHostToDevice)
///
///  @param  allEdgesDevice           GPU address of the allEdges struct on device memory.
///  @param  allEdgesDeviceProps      GPU address of the allDynamicSTDPSSynapses struct on device memory.
///  @param  numVertices                  Number of vertices.
///  @param  maxEdgesPerVertex        Maximum number of synapses per neuron.
void AllDynamicSTDPSynapses::copyHostToDevice( void* allEdgesDevice, AllDynamicSTDPSynapsesDeviceProperties& allEdgesDeviceProps, int numVertices, int maxEdgesPerVertex ) { // copy everything necessary 
        AllSTDPSynapses::copyHostToDevice( allEdgesDevice, allEdgesDeviceProps, numVertices, maxEdgesPerVertex );

        BGSIZE maxTotalSynapses = maxEdgesPerVertex * numVertices;
        
        HANDLE_ERROR( cudaMemcpy ( allEdgesDeviceProps.lastSpike_, lastSpike_,
                maxTotalSynapses * sizeof( uint64_t ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allEdgesDeviceProps.r_, r_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allEdgesDeviceProps.u_, u_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allEdgesDeviceProps.D_, D_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allEdgesDeviceProps.U_, U_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allEdgesDeviceProps.F_, F_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
}

///  Copy all synapses' data from device to host.
///
///  @param  allEdgesDevice  GPU address of the AllDynamicSTDPSynapsesDeviceProperties struct 
///                             on device memory.
void AllDynamicSTDPSynapses::copyEdgeDeviceToHost( void* allEdgesDevice ) {
	// copy everything necessary
	AllDynamicSTDPSynapsesDeviceProperties allEdges;

        HANDLE_ERROR( cudaMemcpy ( &allEdges, allEdgesDevice, sizeof( AllDynamicSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	copyDeviceToHost( allEdges );
}

///  Copy all synapses' data from device to host.
///  (Helper function of copyEdgeDeviceToHost)
///
///  @param  allEdgesDevice           GPU address of the allEdges struct on device memory.
///  @param  allEdgesDeviceProps      GPU address of the allDynamicSTDPSSynapses struct on device memory.
///  @param  numVertices                  Number of vertices.
///  @param  maxEdgesPerVertex        Maximum number of synapses per neuron.
void AllDynamicSTDPSynapses::copyDeviceToHost( AllDynamicSTDPSynapsesDeviceProperties& allEdgesDeviceProps ) {
        AllSTDPSynapses::copyDeviceToHost( allEdgesDeviceProps ) ;

	int numVertices = Simulator::getInstance().getTotalVertices();
	BGSIZE maxTotalSynapses = Simulator::getInstance().getMaxEdgesPerVertex() * numVertices;

        HANDLE_ERROR( cudaMemcpy ( lastSpike_, allEdgesDeviceProps.lastSpike_,
                maxTotalSynapses * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( r_, allEdgesDeviceProps.r_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( u_, allEdgesDeviceProps.u_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( D_, allEdgesDeviceProps.D_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( U_, allEdgesDeviceProps.U_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( F_, allEdgesDeviceProps.F_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
}

///  Set synapse class ID defined by enumClassSynapses for the caller's Synapse class.
///  The class ID will be set to classSynapses_d in device memory,
///  and the classSynapses_d will be referred to call a device function for the
///  particular synapse class.
///  Because we cannot use virtual function (Polymorphism) in device functions,
///  we use this scheme.
///  Note: we used to use a function pointer; however, it caused the growth_cuda crash
///  (see issue#137).
void AllDynamicSTDPSynapses::setEdgeClassID()
{
    enumClassSynapses classSynapses_h = classAllDynamicSTDPSynapses;

    HANDLE_ERROR( cudaMemcpyToSymbol(classSynapses_d, &classSynapses_h, sizeof(enumClassSynapses)) );
}

///  Prints GPU SynapsesProps data.
///   
///  @param  allEdgesDeviceProps   GPU address of the corresponding SynapsesDeviceProperties struct on device memory.
void AllDynamicSTDPSynapses::printGPUEdgesProps( void* allEdgesDeviceProps ) const
{
    AllDynamicSTDPSynapsesDeviceProperties allSynapsesProps;

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
        BGFLOAT *tauspost_Print = new BGFLOAT[size];
        BGFLOAT *tauspre_Print = new BGFLOAT[size];
        BGFLOAT *taupos_Print = new BGFLOAT[size];
        BGFLOAT *tauneg_Print = new BGFLOAT[size];
        BGFLOAT *STDPgap_Print = new BGFLOAT[size];
        BGFLOAT *Wex_Print = new BGFLOAT[size];
        BGFLOAT *Aneg_Print = new BGFLOAT[size];
        BGFLOAT *Apos_Print = new BGFLOAT[size];
        BGFLOAT *mupos_Print = new BGFLOAT[size];
        BGFLOAT *muneg_Print = new BGFLOAT[size];
        bool *useFroemkeDanSTDP_Print = new bool[size];

        uint64_t *lastSpikePrint = new uint64_t[size];
        BGFLOAT *rPrint = new BGFLOAT[size];
        BGFLOAT *uPrint = new BGFLOAT[size];
        BGFLOAT *DPrint = new BGFLOAT[size];
        BGFLOAT *UPrint = new BGFLOAT[size];
        BGFLOAT *FPrint = new BGFLOAT[size];

        // copy everything
        HANDLE_ERROR( cudaMemcpy ( &allSynapsesProps, allEdgesDeviceProps, sizeof( AllDynamicSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );
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
        HANDLE_ERROR( cudaMemcpy ( totalDelayPrint, allSynapsesProps.totalDelay_,size * sizeof( int ), cudaMemcpyDeviceToHost ) );

        HANDLE_ERROR( cudaMemcpy ( totalDelayPostPrint, allSynapsesProps.totalDelayPost_, size * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tauspost_Print, allSynapsesProps.tauspost_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tauspre_Print, allSynapsesProps.tauspre_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( taupos_Print, allSynapsesProps.taupos_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tauneg_Print, allSynapsesProps.tauneg_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( STDPgap_Print, allSynapsesProps.STDPgap_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( Wex_Print, allSynapsesProps.Wex_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( Aneg_Print, allSynapsesProps.Aneg_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( Apos_Print, allSynapsesProps.Apos_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( mupos_Print, allSynapsesProps.mupos_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( muneg_Print, allSynapsesProps.muneg_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( useFroemkeDanSTDP_Print, allSynapsesProps.useFroemkeDanSTDP_, size * sizeof( bool ), cudaMemcpyDeviceToHost ) );

        HANDLE_ERROR( cudaMemcpy ( lastSpikePrint, allSynapsesProps.lastSpike_, size * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( rPrint, allSynapsesProps.r_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( uPrint, allSynapsesProps.u_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( DPrint, allSynapsesProps.D_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( UPrint, allSynapsesProps.U_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( FPrint, allSynapsesProps.F_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );

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
                cout << " GPU tauspost_: " << tauspost_Print[i];
                cout << " GPU tauspre_: " << tauspre_Print[i];
                cout << " GPU taupos_: " << taupos_Print[i];
                cout << " GPU tauneg_: " << tauneg_Print[i];
                cout << " GPU STDPgap_: " << STDPgap_Print[i];
                cout << " GPU Wex_: " << Wex_Print[i];
                cout << " GPU Aneg_: " << Aneg_Print[i];
                cout << " GPU Apos_: " << Apos_Print[i];
                cout << " GPU mupos_: " << mupos_Print[i];
                cout << " GPU muneg_: " << muneg_Print[i];
                cout << " GPU useFroemkeDanSTDP_: " << useFroemkeDanSTDP_Print[i];

                cout << " GPU lastSpike: " << lastSpikePrint[i];
                cout << " GPU r: " << rPrint[i];
                cout << " GPU u: " << uPrint[i];
                cout << " GPU D: " << DPrint[i];
                cout << " GPU U: " << UPrint[i];
                cout << " GPU F: " << FPrint[i] << endl;
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
        delete[] tauspost_Print;
        delete[] tauspre_Print;
        delete[] taupos_Print;
        delete[] tauneg_Print;
        delete[] STDPgap_Print;
        delete[] Wex_Print;
        delete[] Aneg_Print;
        delete[] Apos_Print;
        delete[] mupos_Print;
        delete[] muneg_Print;
        delete[] useFroemkeDanSTDP_Print;
        totalDelayPostPrint = nullptr;
        tauspost_Print = nullptr;
        tauspre_Print = nullptr;
        taupos_Print = nullptr;
        tauneg_Print = nullptr;
        STDPgap_Print = nullptr;
        Wex_Print = nullptr;
        Aneg_Print = nullptr;
        Apos_Print = nullptr;
        mupos_Print = nullptr;
        muneg_Print = nullptr;
        useFroemkeDanSTDP_Print = nullptr;

        delete[] lastSpikePrint;
        delete[] rPrint;
        delete[] uPrint;
        delete[] DPrint;
        delete[] UPrint;
        delete[] FPrint;
        lastSpikePrint = nullptr;
        rPrint = nullptr;
        uPrint = nullptr;
        DPrint = nullptr;
        UPrint = nullptr;
        FPrint = nullptr;
    }
}

