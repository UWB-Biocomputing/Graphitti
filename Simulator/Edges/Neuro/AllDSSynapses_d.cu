/**
 * @file AllDSSynapses_d.cu
 *
 * @ingroup Simulator/Edges
 * 
 * @brief A  container of all DS synapse data
 */

#include "AllSynapsesDeviceFuncs.h"
#include "AllDSSynapses.h"
#include "GPUModel.h"
#include "Simulator.h"
#include "Book.h"

///  Allocate GPU memories to store all synapses' states,
///  and copy them from host to GPU memory.
///
///  @param  allEdgesDevice  GPU address of the AllDSSynapsesDeviceProperties struct 
///                             on device memory.
void AllDSSynapses::allocEdgeDeviceStruct ( void** allEdgesDevice ) {
	allocEdgeDeviceStruct(allEdgesDevice, Simulator::getInstance().getTotalVertices(), Simulator::getInstance().getMaxEdgesPerVertex());
}

///  Allocate GPU memories to store all synapses' states,
///  and copy them from host to GPU memory.
///
///  @param  allEdgesDevice     GPU address of the AllDSSynapsesDeviceProperties struct 
///                                on device memory.
///  @param  numVertices            Number of vertices.
///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
void AllDSSynapses::allocEdgeDeviceStruct( void** allEdgesDevice, int numVertices, int maxEdgesPerVertex ) {
	AllDSSynapsesDeviceProperties allEdges;

	allocDeviceStruct( allEdges, numVertices, maxEdgesPerVertex );

	HANDLE_ERROR( cudaMalloc ( allEdgesDevice, sizeof( AllDSSynapsesDeviceProperties ) ) );
	HANDLE_ERROR( cudaMemcpy ( *allEdgesDevice, &allEdges, sizeof( AllDSSynapsesDeviceProperties ), cudaMemcpyHostToDevice ) );
}

///  Allocate GPU memories to store all synapses' states,
///  and copy them from host to GPU memory.
///  (Helper function of allocEdgeDeviceStruct)
///
///  @param  allEdgesDevice     GPU address of the AllDSSynapsesDeviceProperties struct 
///                                on device memory.
///  @param  numVertices            Number of vertices.
///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
void AllDSSynapses::allocDeviceStruct( AllDSSynapsesDeviceProperties &allEdges, int numVertices, int maxEdgesPerVertex ) {
        AllSpikingSynapses::allocDeviceStruct( allEdges, numVertices, maxEdgesPerVertex );

        BGSIZE maxTotalSynapses = maxEdgesPerVertex * numVertices;

        HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdges.lastSpike_, maxTotalSynapses * sizeof( uint64_t ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdges.r_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdges.u_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdges.D_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdges.U_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allEdges.F_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
}

///  Delete GPU memories.
///
///  @param  allEdgesDevice  GPU address of the AllDSSynapsesDeviceProperties struct 
///                             on device memory.
void AllDSSynapses::deleteEdgeDeviceStruct( void* allEdgesDevice ) {
	AllDSSynapsesDeviceProperties allEdges;

	HANDLE_ERROR( cudaMemcpy ( &allEdges, allEdgesDevice, sizeof( AllDSSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allEdges );

	HANDLE_ERROR( cudaFree( allEdgesDevice ) );
}

///  Delete GPU memories.
///  (Helper function of deleteEdgeDeviceStruct)
///
///  @param  allEdgesDeviceProps  GPU address of the AllDSSynapsesDeviceProperties struct 
///                                  on device memory.
void AllDSSynapses::deleteDeviceStruct( AllDSSynapsesDeviceProperties& allEdgesDeviceProps ) {
        HANDLE_ERROR( cudaFree( allEdgesDeviceProps.lastSpike_ ) );
	HANDLE_ERROR( cudaFree( allEdgesDeviceProps.r_ ) );
	HANDLE_ERROR( cudaFree( allEdgesDeviceProps.u_ ) );
	HANDLE_ERROR( cudaFree( allEdgesDeviceProps.D_ ) );
	HANDLE_ERROR( cudaFree( allEdgesDeviceProps.U_ ) );
	HANDLE_ERROR( cudaFree( allEdgesDeviceProps.F_ ) );

        AllSpikingSynapses::deleteDeviceStruct( allEdgesDeviceProps );
}

///  Copy all synapses' data from host to device.
///
///  @param  allEdgesDevice  GPU address of the AllDSSynapsesDeviceProperties struct 
///                            on device memory.
void AllDSSynapses::copyEdgeHostToDevice( void* allEdgesDevice) { // copy everything necessary
	copyEdgeHostToDevice(allEdgesDevice, Simulator::getInstance().getTotalVertices(), Simulator::getInstance().getMaxEdgesPerVertex());	
}

///  Copy all synapses' data from host to device.
///
///  @param  allEdgesDevice     GPU address of the AllDSSynapsesDeviceProperties struct 
///                                on device memory.
///  @param  numVertices            Number of vertices.
///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
void AllDSSynapses::copyEdgeHostToDevice( void* allEdgesDevice, int numVertices, int maxEdgesPerVertex ) { // copy everything necessary
	AllDSSynapsesDeviceProperties allEdges;

        HANDLE_ERROR( cudaMemcpy ( &allEdges, allEdgesDevice, sizeof( AllDSSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );
	copyHostToDevice( allEdgesDevice, allEdges, numVertices, maxEdgesPerVertex );	
}

///  Copy all synapses' data from host to device.
///  (Helper function of copyEdgeHostToDevice)
///
///  @param  allEdgesDevice     GPU address of the AllDSSynapsesDeviceProperties struct 
///                                on device memory.
///  @param  numVertices            Number of vertices.
///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
void AllDSSynapses::copyHostToDevice( void* allEdgesDevice, AllDSSynapsesDeviceProperties& allEdgesDeviceProps, int numVertices, int maxEdgesPerVertex ) { // copy everything necessary 
        AllSpikingSynapses::copyHostToDevice( allEdgesDevice, allEdgesDeviceProps, numVertices, maxEdgesPerVertex );

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
///  @param  allEdgesDevice  GPU address of the AllDSSynapsesDeviceProperties struct 
///                             on device memory.
void AllDSSynapses::copyEdgeDeviceToHost( void* allEdgesDevice ) {
	// copy everything necessary
	AllDSSynapsesDeviceProperties allEdgesDeviceProps;

        HANDLE_ERROR( cudaMemcpy ( &allEdgesDeviceProps, allEdgesDevice, sizeof( AllDSSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	copyDeviceToHost( allEdgesDeviceProps );
}

///  Copy all synapses' data from device to host.
///  (Helper function of copyEdgeDeviceToHost)
///
///  @param  allEdgesDeviceProps     GPU address of the AllDSSynapsesDeviceProperties struct 
///                                     on device memory.
///  @param  numVertices                 Number of vertices.
///  @param  maxEdgesPerVertex       Maximum number of synapses per neuron.
void AllDSSynapses::copyDeviceToHost( AllDSSynapsesDeviceProperties& allEdgesDeviceProps ) {
        AllSpikingSynapses::copyDeviceToHost( allEdgesDeviceProps ) ;

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
void AllDSSynapses::setEdgeClassID()
{
    enumClassSynapses classSynapses_h = classAllDSSynapses;

    HANDLE_ERROR( cudaMemcpyToSymbol ( classSynapses_d, &classSynapses_h, sizeof(enumClassSynapses) ) );
}

///  Prints GPU SynapsesProps data.
/// 
///  @param  allEdgesDeviceProps   GPU address of the corresponding SynapsesDeviceProperties struct on device memory.
void AllDSSynapses::printGPUEdgesProps( void* allEdgesDeviceProps ) const
{
    AllDSSynapsesDeviceProperties allSynapsesProps;

    //allocate print out data members
    BGSIZE size = Simulator::getInstance().getMaxEdgesPerVertex() * countVertices_;
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

        uint64_t *lastSpikePrint = new uint64_t[size];
        BGFLOAT *rPrint = new BGFLOAT[size];
        BGFLOAT *uPrint = new BGFLOAT[size];
        BGFLOAT *DPrint = new BGFLOAT[size];
        BGFLOAT *UPrint = new BGFLOAT[size];
        BGFLOAT *FPrint = new BGFLOAT[size];


        // copy everything
        HANDLE_ERROR( cudaMemcpy ( &allSynapsesProps, allEdgesDeviceProps, sizeof( AllDSSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );
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


        HANDLE_ERROR( cudaMemcpy ( lastSpikePrint, allSynapsesProps.lastSpike_, size * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( rPrint, allSynapsesProps.r_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( uPrint, allSynapsesProps.u_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( DPrint, allSynapsesProps.D_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( UPrint, allSynapsesProps.U_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( FPrint, allSynapsesProps.F_, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );


        for(int i = 0; i < size; i++) {
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
