#include "ConnGrowth.h"
#include "AllSpikingSynapses.h"
#include "AllSynapsesDeviceFuncs.h"
#include "Simulator.h"
#include "Book.h"

/*
 *  Update the weights of the Synapses in the simulation. To be clear,
 *  iterates through all source and destination neurons and updates their
 *  synaptic strengths from the weight matrix.
 *  Note: Platform Dependent.
 *
 *  @param  numNeurons         number of neurons to update.
 *  @param  neurons             the AllNeurons object.
 *  @param  synapses            the AllSynapses object.
 *  @param  allNeuronsDevice_  Reference to the allNeurons struct in device memory.
 *  @param  allSynapseDevice_ Reference to the allSynapses struct in device memory.
 *  @param  layout              the Layout object.
 */
void ConnGrowth::updateSynapsesWeights(const int numNeurons, IAllNeurons &neurons, IAllSynapses &synapses, AllSpikingNeuronsDeviceProperties* allNeuronsDevice_, AllSpikingSynapsesDeviceProperties* allSynapseDevice_, Layout *layout)
{
        Simulator &simulator = Simulator::getInstance();
        // For now, we just set the weights to equal the areas. We will later
        // scale it and set its sign (when we index and get its sign).
        (*W_) = (*area_);

        BGFLOAT deltaT = simulator.getDeltaT();

        // CUDA parameters
        const int threadsPerBlock = 256;
        int blocksPerGrid;

        // allocate device memories
        BGSIZE W_d_size = simulator.getTotalNeurons() * simulator.getTotalNeurons() * sizeof (BGFLOAT);
        BGFLOAT* W_h = new BGFLOAT[W_d_size];
        BGFLOAT* W_d;
        HANDLE_ERROR( cudaMalloc ( ( void ** ) &W_d, W_d_size ) );

        neuronType* neuronTypeMapD;
        HANDLE_ERROR( cudaMalloc( ( void ** ) &neuronTypeMapD, simulator.getTotalNeurons() * sizeof( neuronType ) ) );

        // copy weight data to the device memory
        for ( int i = 0 ; i < simulator.getTotalNeurons(); i++ )
                for ( int j = 0; j < simulator.getTotalNeurons(); j++ )
                        W_h[i * simulator.getTotalNeurons() + j] = (*W_)(i, j);

        HANDLE_ERROR( cudaMemcpy ( W_d, W_h, W_d_size, cudaMemcpyHostToDevice ) );

        HANDLE_ERROR( cudaMemcpy ( neuronTypeMapD, layout->neuronTypeMap_, simulator.getTotalNeurons() * sizeof( neuronType ), cudaMemcpyHostToDevice ) );

        blocksPerGrid = ( simulator.getTotalNeurons() + threadsPerBlock - 1 ) / threadsPerBlock;
        updateSynapsesWeightsDevice <<< blocksPerGrid, threadsPerBlock >>> ( simulator.getTotalNeurons(), deltaT, W_d, simulator.getMaxSynapsesPerNeuron(), allNeuronsDevice_, allSynapseDevice_, neuronTypeMapD );

        // free memories
        HANDLE_ERROR( cudaFree( W_d ) );
        delete[] W_h;

        HANDLE_ERROR( cudaFree( neuronTypeMapD ) );

        // copy device synapse count to host memory
        synapses.copyDeviceSynapseCountsToHost(allSynapseDevice_);
        // copy device synapse summation coordinate to host memory
        synapses.copyDeviceSynapseSumIdxToHost(allSynapseDevice_);
}
