/*
 * AllDSSynapses_d.cu
 *
 */

 #include "AllSynapsesDeviceFuncs.h"
#include "AllDSSynapses.h"
#include "GPUSpikingModel.h"
#include "Simulator.h"
#include "Book.h"

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDevice  GPU address of the AllDSSynapsesDeviceProperties struct 
 *                             on device memory.
 */
void AllDSSynapses::allocSynapseDeviceStruct ( void** allSynapsesDevice ) {
	allocSynapseDeviceStruct(allSynapsesDevice, Simulator::getInstance().getTotalNeurons(), Simulator::getInstance().getMaxSynapsesPerNeuron());
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDevice     GPU address of the AllDSSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  numNeurons            Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDSSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, int numNeurons, int maxSynapsesPerNeuron ) {
	AllDSSynapsesDeviceProperties allSynapses;

	allocDeviceStruct( allSynapses, numNeurons, maxSynapsesPerNeuron );

	HANDLE_ERROR( cudaMalloc ( allSynapsesDevice, sizeof( AllDSSynapsesDeviceProperties ) ) );
	HANDLE_ERROR( cudaMemcpy ( *allSynapsesDevice, &allSynapses, sizeof( AllDSSynapsesDeviceProperties ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *  (Helper function of allocSynapseDeviceStruct)
 *
 *  @param  allSynapsesDevice     GPU address of the AllDSSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  numNeurons            Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDSSynapses::allocDeviceStruct( AllDSSynapsesDeviceProperties &allSynapses, int numNeurons, int maxSynapsesPerNeuron ) {
        AllSpikingSynapses::allocDeviceStruct( allSynapses, numNeurons, maxSynapsesPerNeuron );

        BGSIZE maxTotalSynapses = maxSynapsesPerNeuron * numNeurons;

        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.lastSpike_, maxTotalSynapses * sizeof( uint64_t ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.r_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.u_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.D_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.U_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.F_, maxTotalSynapses * sizeof( BGFLOAT ) ) );
}

/*
 *  Delete GPU memories.
 *
 *  @param  allSynapsesDevice  GPU address of the AllDSSynapsesDeviceProperties struct 
 *                             on device memory.
 */
void AllDSSynapses::deleteSynapseDeviceStruct( void* allSynapsesDevice ) {
	AllDSSynapsesDeviceProperties allSynapses;

	HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllDSSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allSynapses );

	HANDLE_ERROR( cudaFree( allSynapsesDevice ) );
}

/*
 *  Delete GPU memories.
 *  (Helper function of deleteSynapseDeviceStruct)
 *
 *  @param  allSynapsesDeviceProps  GPU address of the AllDSSynapsesDeviceProperties struct 
 *                                  on device memory.
 */
void AllDSSynapses::deleteDeviceStruct( AllDSSynapsesDeviceProperties& allSynapsesDeviceProps ) {
        HANDLE_ERROR( cudaFree( allSynapsesDeviceProps.lastSpike_ ) );
	HANDLE_ERROR( cudaFree( allSynapsesDeviceProps.r_ ) );
	HANDLE_ERROR( cudaFree( allSynapsesDeviceProps.u_ ) );
	HANDLE_ERROR( cudaFree( allSynapsesDeviceProps.D_ ) );
	HANDLE_ERROR( cudaFree( allSynapsesDeviceProps.U_ ) );
	HANDLE_ERROR( cudaFree( allSynapsesDeviceProps.F_ ) );

        AllSpikingSynapses::deleteDeviceStruct( allSynapsesDeviceProps );
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesDevice  GPU address of the AllDSSynapsesDeviceProperties struct 
 *                             on device memory.
 */
void AllDSSynapses::copySynapseHostToDevice( void* allSynapsesDevice) { // copy everything necessary
	copySynapseHostToDevice(allSynapsesDevice, Simulator::getInstance().getTotalNeurons(), Simulator::getInstance().getMaxSynapsesPerNeuron());	
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesDevice     GPU address of the AllDSSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  numNeurons            Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDSSynapses::copySynapseHostToDevice( void* allSynapsesDevice, int numNeurons, int maxSynapsesPerNeuron ) { // copy everything necessary
	AllDSSynapsesDeviceProperties allSynapses;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllDSSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );
	copyHostToDevice( allSynapsesDevice, allSynapses, numNeurons, maxSynapsesPerNeuron );	
}

/*
 *  Copy all synapses' data from host to device.
 *  (Helper function of copySynapseHostToDevice)
 *
 *  @param  allSynapsesDevice     GPU address of the AllDSSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  numNeurons            Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDSSynapses::copyHostToDevice( void* allSynapsesDevice, AllDSSynapsesDeviceProperties& allSynapsesDeviceProps, int numNeurons, int maxSynapsesPerNeuron ) { // copy everything necessary 
        AllSpikingSynapses::copyHostToDevice( allSynapsesDevice, allSynapsesDeviceProps, numNeurons, maxSynapsesPerNeuron );

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

/*
 *  Copy all synapses' data from device to host.
 *
 *  @param  allSynapsesDevice  GPU address of the AllDSSynapsesDeviceProperties struct 
 *                             on device memory.
 */
void AllDSSynapses::copySynapseDeviceToHost( void* allSynapsesDevice ) {
	// copy everything necessary
	AllDSSynapsesDeviceProperties allSynapsesDeviceProps;

        HANDLE_ERROR( cudaMemcpy ( &allSynapsesDeviceProps, allSynapsesDevice, sizeof( AllDSSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	copyDeviceToHost( allSynapsesDeviceProps );
}

/*
 *  Copy all synapses' data from device to host.
 *  (Helper function of copySynapseDeviceToHost)
 *
 *  @param  allSynapsesDeviceProps     GPU address of the AllDSSynapsesDeviceProperties struct 
 *                                     on device memory.
 *  @param  numNeurons                 Number of neurons.
 *  @param  maxSynapsesPerNeuron       Maximum number of synapses per neuron.
 */
void AllDSSynapses::copyDeviceToHost( AllDSSynapsesDeviceProperties& allSynapsesDeviceProps ) {
        AllSpikingSynapses::copyDeviceToHost( allSynapsesDeviceProps ) ;

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
void AllDSSynapses::setSynapseClassID()
{
    enumClassSynapses classSynapses_h = classAllDSSynapses;

    HANDLE_ERROR( cudaMemcpyToSymbol ( classSynapses_d, &classSynapses_h, sizeof(enumClassSynapses) ) );
}

/*
 *  Prints GPU SynapsesProps data.
 * 
 *  @param  allSynapsesDeviceProps   GPU address of the corresponding SynapsesDeviceProperties struct on device memory.
 */
void AllDSSynapses::printGPUSynapsesProps( void* allSynapsesDeviceProps ) const
{
    AllDSSynapsesDeviceProperties allSynapsesProps;

    //allocate print out data members
    BGSIZE size = Simulator::getInstance().getMaxSynapsesPerNeuron() * countNeurons_;
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

        uint64_t *lastSpikePrint = new uint64_t[size];
        BGFLOAT *rPrint = new BGFLOAT[size];
        BGFLOAT *uPrint = new BGFLOAT[size];
        BGFLOAT *DPrint = new BGFLOAT[size];
        BGFLOAT *UPrint = new BGFLOAT[size];
        BGFLOAT *FPrint = new BGFLOAT[size];


        // copy everything
        HANDLE_ERROR( cudaMemcpy ( &allSynapsesProps, allSynapsesDeviceProps, sizeof( AllDSSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );
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
