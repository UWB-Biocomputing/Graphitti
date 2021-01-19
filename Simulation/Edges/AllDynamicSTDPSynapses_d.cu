/**
 * @file AllDynamicSTDPSynapses_d.cu
 *
 * @ingroup Simulation/Edges
 * 
 * @brief
 */

#include "AllDynamicSTDPSynapses.h"
#include "AllSynapsesDeviceFuncs.h"
#include "Book.h"
#include "Simulator.h"

///  Allocate GPU memories to store all synapses' states,
///  and copy them from host to GPU memory.
///
///  @param  allSynapsesDevice  GPU address of the AllDynamicSTDPSynapsesDeviceProperties struct 
///                             on device memory.
void AllDynamicSTDPSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice ) {
	allocSynapseDeviceStruct( allSynapsesDevice, Simulator::getInstance().getTotalNeurons(), Simulator::getInstance().getMaxSynapsesPerNeuron() );
}

///  Allocate GPU memories to store all synapses' states,
///  and copy them from host to GPU memory.
///
///  @param  allSynapsesDevice     GPU address of the AllDynamicSTDPSynapsesDeviceProperties struct 
///                                on device memory.
///  @param  numNeurons            Number of neurons.
///  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
void AllDynamicSTDPSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, int numNeurons, int maxSynapsesPerNeuron ) {
	AllDynamicSTDPSynapsesDeviceProperties allSynapses;

	allocDeviceStruct( allSynapses, numNeurons, maxSynapsesPerNeuron );

	HANDLE_ERROR( cudaMalloc( allSynapsesDevice, sizeof( AllDynamicSTDPSynapsesDeviceProperties ) ) );
	HANDLE_ERROR( cudaMemcpy ( *allSynapsesDevice, &allSynapses, sizeof( AllDynamicSTDPSynapsesDeviceProperties ), cudaMemcpyHostToDevice ) );
}

///  Allocate GPU memories to store all synapses' states,
///  and copy them from host to GPU memory.
///  (Helper function of allocSynapseDeviceStruct)
///
///  @param  allSynapsesDeviceProps      GPU address of the AllDynamicSTDPSynapsesDeviceProperties struct 
///                                      on device memory.
///  @param  numNeurons                  Number of neurons.
///  @param  maxSynapsesPerNeuron        Maximum number of synapses per neuron.
void AllDynamicSTDPSynapses::allocDeviceStruct( AllDynamicSTDPSynapsesDeviceProperties &allSynapsesDeviceProps, int numNeurons, int maxSynapsesPerNeuron ) {
        AllSTDPSynapses::allocDeviceStruct( allSynapsesDeviceProps, numNeurons, maxSynapsesPerNeuron );

        BGSIZE maxTotalSynapses = maxSynapsesPerNeuron * numNeurons;

        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDeviceProps.lastSpike_, maxTotalSynapses * sizeof( uint64_t ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDeviceProps.r_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDeviceProps.u_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDeviceProps.D_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDeviceProps.U_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDeviceProps.F_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
}

///  Delete GPU memories.
///
///  @param  allSynapsesDevice  GPU address of the AllDynamicSTDPSynapsesDeviceProperties struct 
///                             on device memory.
void AllDynamicSTDPSynapses::deleteSynapseDeviceStruct( void* allSynapsesDevice ) {
	AllDynamicSTDPSynapsesDeviceProperties allSynapses;

	HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllDynamicSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allSynapses );

	HANDLE_ERROR( cudaFree( allSynapsesDevice ) );
}

///  Delete GPU memories.
///  (Helper function of deleteSynapseDeviceStruct)
///
///  @param  allSynapsesDeviceProps  GPU address of the AllDynamicSTDPSynapsesDeviceProperties struct 
///                                  on device memory.
void AllDynamicSTDPSynapses::deleteDeviceStruct( AllDynamicSTDPSynapsesDeviceProperties& allSynapsesDeviceProps ) {
        HANDLE_ERROR( cudaFree( allSynapsesDeviceProps.lastSpike_ ) );
	HANDLE_ERROR( cudaFree( allSynapsesDeviceProps.r_ ) );
	HANDLE_ERROR( cudaFree( allSynapsesDeviceProps.u_ ) );
	HANDLE_ERROR( cudaFree( allSynapsesDeviceProps.D_ ) );
	HANDLE_ERROR( cudaFree( allSynapsesDeviceProps.U_ ) );
	HANDLE_ERROR( cudaFree( allSynapsesDeviceProps.F_ ) );

        AllSTDPSynapses::deleteDeviceStruct( allSynapsesDeviceProps );
}

///  Copy all synapses' data from host to device.
///
///  @param  allSynapsesDevice  GPU address of the AllDynamicSTDPSynapsesDeviceProperties struct 
///                             on device memory.
void AllDynamicSTDPSynapses::copySynapseHostToDevice( void* allSynapsesDevice ) { // copy everything necessary
	copySynapseHostToDevice( allSynapsesDevice, Simulator::getInstance().getTotalNeurons(), Simulator::getInstance().getMaxSynapsesPerNeuron() );	
}

///  Copy all synapses' data from host to device.
///
///  @param  allSynapsesDevice     GPU address of the AllDynamicSTDPSynapsesDeviceProperties struct 
///                                on device memory.
///  @param  numNeurons            Number of neurons.
///  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
void AllDynamicSTDPSynapses::copySynapseHostToDevice( void* allSynapsesDevice, int numNeurons, int maxSynapsesPerNeuron ) { // copy everything necessary
	AllDynamicSTDPSynapsesDeviceProperties allSynapses;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllDynamicSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	copyHostToDevice( allSynapsesDevice, allSynapses, numNeurons, maxSynapsesPerNeuron );	
}

///  Copy all synapses' data from host to device.
///  (Helper function of copySynapseHostToDevice)
///
///  @param  allSynapsesDevice           GPU address of the allSynapses struct on device memory.
///  @param  allSynapsesDeviceProps      GPU address of the allDynamicSTDPSSynapses struct on device memory.
///  @param  numNeurons                  Number of neurons.
///  @param  maxSynapsesPerNeuron        Maximum number of synapses per neuron.
void AllDynamicSTDPSynapses::copyHostToDevice( void* allSynapsesDevice, AllDynamicSTDPSynapsesDeviceProperties& allSynapsesDeviceProps, int numNeurons, int maxSynapsesPerNeuron ) { // copy everything necessary 
        AllSTDPSynapses::copyHostToDevice( allSynapsesDevice, allSynapsesDeviceProps, numNeurons, maxSynapsesPerNeuron );

        BGSIZE maxTotalSynapses = maxSynapsesPerNeuron * numNeurons;
        
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.lastSpike_, lastSpike_,
                maxTotalSynapses * sizeof( uint64_t ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.r_, r_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.u_, u_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.D_, D_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.U_, U_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDeviceProps.F_, F_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
}

///  Copy all synapses' data from device to host.
///
///  @param  allSynapsesDevice  GPU address of the AllDynamicSTDPSynapsesDeviceProperties struct 
///                             on device memory.
void AllDynamicSTDPSynapses::copySynapseDeviceToHost( void* allSynapsesDevice ) {
	// copy everything necessary
	AllDynamicSTDPSynapsesDeviceProperties allSynapses;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllDynamicSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	copyDeviceToHost( allSynapses );
}

///  Copy all synapses' data from device to host.
///  (Helper function of copySynapseDeviceToHost)
///
///  @param  allSynapsesDevice           GPU address of the allSynapses struct on device memory.
///  @param  allSynapsesDeviceProps      GPU address of the allDynamicSTDPSSynapses struct on device memory.
///  @param  numNeurons                  Number of neurons.
///  @param  maxSynapsesPerNeuron        Maximum number of synapses per neuron.
void AllDynamicSTDPSynapses::copyDeviceToHost( AllDynamicSTDPSynapsesDeviceProperties& allSynapsesDeviceProps ) {
        AllSTDPSynapses::copyDeviceToHost( allSynapsesDeviceProps ) ;

	int numNeurons = Simulator::getInstance().getTotalNeurons();
	BGSIZE maxTotalSynapses = Simulator::getInstance().getMaxSynapsesPerNeuron() * numNeurons;

        HANDLE_ERROR( cudaMemcpy ( lastSpike_, allSynapsesDeviceProps.lastSpike_,
                maxTotalSynapses * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( r_, allSynapsesDeviceProps.r_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( u_, allSynapsesDeviceProps.u_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( D_, allSynapsesDeviceProps.D_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( U_, allSynapsesDeviceProps.U_,
                maxTotalSynapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( F_, allSynapsesDeviceProps.F_,
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
void AllDynamicSTDPSynapses::setSynapseClassID()
{
    enumClassSynapses classSynapses_h = classAllDynamicSTDPSynapses;

    HANDLE_ERROR( cudaMemcpyToSymbol(classSynapses_d, &classSynapses_h, sizeof(enumClassSynapses)) );
}

///  Prints GPU SynapsesProps data.
///   
///  @param  allSynapsesDeviceProps   GPU address of the corresponding SynapsesDeviceProperties struct on device memory.
void AllDynamicSTDPSynapses::printGPUSynapsesProps( void* allSynapsesDeviceProps ) const
{
    AllDynamicSTDPSynapsesDeviceProperties allSynapsesProps;

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
        HANDLE_ERROR( cudaMemcpy ( &allSynapsesProps, allSynapsesDeviceProps, sizeof( AllDynamicSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );
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
        totalDelayPostPrint = NULL;
        tauspost_Print = NULL;
        tauspre_Print = NULL;
        taupos_Print = NULL;
        tauneg_Print = NULL;
        STDPgap_Print = NULL;
        Wex_Print = NULL;
        Aneg_Print = NULL;
        Apos_Print = NULL;
        mupos_Print = NULL;
        muneg_Print = NULL;
        useFroemkeDanSTDP_Print = NULL;

        delete[] lastSpikePrint;
        delete[] rPrint;
        delete[] uPrint;
        delete[] DPrint;
        delete[] UPrint;
        delete[] FPrint;
        lastSpikePrint = NULL;
        rPrint = NULL;
        uPrint = NULL;
        DPrint = NULL;
        UPrint = NULL;
        FPrint = NULL;
    }
}

