/*
 * @file GpuSInputPoisson.cu
 *
 * @ingroup Simulator/Utils/Inputs
 *
 * @brief A class that performs stimulus input (implementation Poisson) on GPU.
 */

#include "curand_kernel.h"
#include "GpuSInputPoisson.h"
#include "Book.h"

/// Memory to save global state for curand.
curandState* devStates_d;

/// constructor
///
/// @param[in] psi       Pointer to the simulation information
/// @param[in] parms     TiXmlElement to examine.
GpuSInputPoisson::GpuSInputPoisson(SimulationInfo* psi, TiXmlElement* parms) : SInputPoisson(psi, parms)
{
}

/// destructor
GpuSInputPoisson::~GpuSInputPoisson()
{
}

/// Initialize data.
///
/// @param[in] psi       Pointer to the simulation information.
void GpuSInputPoisson::init(SimulationInfo* psi)
{
    SInputPoisson::init(psi);

    if (fSInput == false)
        return;

    // allocate GPU device memory and copy values
    allocDeviceValues(psi->model, psi, nISIs);

    // CUDA parameters
    int vertex_count = psi->totalVertices;
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( vertex_count + threadsPerBlock - 1 ) / threadsPerBlock;

    // setup seeds
    setupSeeds <<< blocksPerGrid, threadsPerBlock >>> ( vertex_count, devStates_d, time(NULL) );
}

/// Terminate process.
///
/// @param[in] psi                Pointer to the simulation information.
void GpuSInputPoisson::term(SimulationInfo* psi)
{
    SInputPoisson::term(psi);

    if (fSInput)
        deleteDeviceValues(psi->model, psi);
}

/// Process input stimulus for each time step.
/// Apply inputs on summationPoint.
///
/// @param[in] psi                Pointer to the simulation information.
void GpuSInputPoisson::inputStimulus(SimulationInfo* psi)
{
    if (fSInput == false)
        return;

    int vertex_count = psi->totalVertices;
    int edge_count = psi->totalVertices;

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( vertex_count + threadsPerBlock - 1 ) / threadsPerBlock;

    // add input spikes to each synapse
    inputStimulusDevice <<< blocksPerGrid, threadsPerBlock >>> ( vertex_count, nISIs_d, masks_d, psi->deltaT, lambda, devStates_d, allEdgesDevice );

    // advance synapses
    advanceSpikingSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( edge_count, edgeIndexMapDevice, g_simulationStep, psi->deltaT, (AllSpikingSynapsesDeviceProperties*)allEdgesDevice );

    // update summation point
    applyI2SummationMap <<< blocksPerGrid, threadsPerBlock >>> ( vertex_count, psi->pSummationMap, allEdgesDevice );
}

/// Allocate GPU device memory and copy values
///
/// @param[in] model      Pointer to the Neural Network Model object.
/// @param[in] psi        Pointer to the simulation information.
/// @param[in] nISIs      Pointer to the interval counter.
void GpuSInputPoisson::allocDeviceValues(IModel* model, SimulationInfo* psi, int *nISIs )
{
    int vertex_count = psi->totalVertices;
    BGSIZE nISIs_d_size = vertex_count * sizeof (int);   // size of shift values

    // Allocate GPU device memory
    HANDLE_ERROR( cudaMalloc ( ( void ** ) &nISIs_d, nISIs_d_size ) );

    // Copy values into device memory
    HANDLE_ERROR( cudaMemcpy ( nISIs_d, nISIs, nISIs_d_size, cudaMemcpyHostToDevice ) );

    // create an input synapse layer
    m_synapses->allocEdgeDeviceStruct( (void **)&allEdgesDevice, vertex_count, 1 ); 
    m_synapses->copyEdgeHostToDevice( allEdgesDevice, vertex_count, 1 );

    const int threadsPerBlock = 256;
    int blocksPerGrid = ( vertex_count + threadsPerBlock - 1 ) / threadsPerBlock;

    initSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( vertex_count, allEdgesDevice, psi->pSummationMap, psi->width, psi->deltaT, weight );

    // allocate memory for curand global state
    HANDLE_ERROR( cudaMalloc ( &devStates_d, vertex_count * sizeof( curandState ) ) );

    // allocate memory for synapse index map and initialize it
    EdgeIndexMap edgeIndexMap;
    BGSIZE* incomingSynapseIndexMap = new BGSIZE[vertex_count];

    BGSIZE syn_i = 0;
    for (int i = 0; i < vertex_count; i++, syn_i++)
    {
        incomingSynapseIndexMap[i] = syn_i;
    }
    HANDLE_ERROR( cudaMalloc( ( void ** ) &edgeIndexMap.incomingSynapseIndexMap, vertex_count * sizeof( BGSIZE ) ) );
    HANDLE_ERROR( cudaMemcpy ( edgeIndexMap.incomingSynapseIndexMap, incomingSynapseIndexMap, vertex_count * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) ); 
    HANDLE_ERROR( cudaMalloc( ( void ** ) &edgeIndexMapDevice, sizeof( EdgeIndexMap ) ) );
    HANDLE_ERROR( cudaMemcpy ( edgeIndexMapDevice, &edgeIndexMap, sizeof( EdgeIndexMap ), cudaMemcpyHostToDevice ) );

    delete[] incomingSynapseIndexMap;

    // allocate memory for masks for stimulus input and initialize it
    HANDLE_ERROR( cudaMalloc ( &masks_d, vertex_count * sizeof( bool ) ) );
    HANDLE_ERROR( cudaMemcpy ( masks_d, masks, vertex_count * sizeof( bool ), cudaMemcpyHostToDevice ) ); 
}

/// Dellocate GPU device memory
///
/// @param[in] model      Pointer to the Neural Network Model object.
/// @param[in] psi        Pointer to the simulation information.
void GpuSInputPoisson::deleteDeviceValues(IModel* model, SimulationInfo* psi )
{
    HANDLE_ERROR( cudaFree( nISIs_d ) );
    HANDLE_ERROR( cudaFree( devStates_d ) );
    HANDLE_ERROR( cudaFree( masks_d ) );

    m_synapses->deleteEdgeDeviceStruct( allEdgesDevice );

    // deallocate memory for synapse index map
    EdgeIndexMap edgeIndexMap;
    HANDLE_ERROR( cudaMemcpy ( &edgeIndexMap, edgeIndexMapDevice, sizeof( EdgeIndexMap ), cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaFree( edgeIndexMap.incomingSynapseIndexMap ) );
    HANDLE_ERROR( cudaFree( edgeIndexMapDevice ) );
}

/// Device code for adding input values to the summation map.
///
/// @param[in] nISIs_d            Pointer to the interval counter.
/// @param[in] masks_d            Pointer to the input stimulus masks.
/// @param[in] deltaT             Time step of the simulation in second.
/// @param[in] lambda             Iinverse firing rate.
/// @param[in] devStates_d        Curand global state
/// @param[in] allEdgesDevice  Pointer to Synapse structures in device memory.
__global__ void inputStimulusDevice( int n, int* nISIs_d, bool* masks_d, BGFLOAT deltaT, BGFLOAT lambda, curandState* devStates_d, AllDSSynapsesDeviceProperties* allEdgesDevice )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n )
        return;

    if (masks_d[idx] == false)
        return;

    BGSIZE iEdg = idx;

    int rnISIs = nISIs_d[idx];    // load the value to a register
    if (--rnISIs <= 0)
    {
        // add a spike
        uint32_t &delay_queue = allEdgesDevice->delayQueue[iEdg];
        int delayIdx = allEdgesDevice->delayIdx[iEdg];
        int ldelayQueue = allEdgesDevice->ldelayQueue[iEdg];
        int total_delay = allEdgesDevice->total_delay[iEdg];

        // Add to spike queue

        // calculate index where to insert the spike into delayQueue
        int idx = delayIdx +  total_delay;
        if ( idx >= ldelayQueue ) {
            idx -= ldelayQueue;
        }

        // set a spike
        //assert( !(delay_queue[0] & (0x1 << idx)) );
        delay_queue |= (0x1 << idx);

        // update interval counter (exponectially distribution ISIs, Poisson)
        curandState localState = devStates_d[idx];

        BGFLOAT isi = -lambda * log(curand_uniform( &localState ));
        // delete isi within refractoriness
        while (curand_uniform( &localState ) <= exp(-(isi*isi)/32))
            isi = -lambda * log(curand_uniform( &localState ));
        // convert isi from msec to steps
        rnISIs = static_cast<int>( (isi / 1000) / deltaT + 0.5 );
        devStates_d[idx] = localState;
    }
    nISIs_d[idx] = rnISIs;
}

/// CUDA code for update summation point
///
/// @param[in] n                  Number of vertices.
/// @param[in] summationPoint_d   SummationPoint
/// @param[in] allEdgesDevice  Pointer to Synapse structures in device memory.
__global__ void applyI2SummationMap( int n, BGFLOAT* summationPoint_d, AllDSSynapsesDeviceProperties* allEdgesDevice ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n )
            return;

    summationPoint_d[idx] += allEdgesDevice->psr[idx];
}

/// CUDA code for setup curand seed
///
/// @param[in] n                  Number of vertices.
/// @param[in] devStates_d        Curand global state
/// @param[in] seed               Seed
__global__ void setupSeeds( int n, curandState* devStates_d, unsigned long seed )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n )
            return;

    curand_init( seed, idx, 0, &devStates_d[idx] );
} 
