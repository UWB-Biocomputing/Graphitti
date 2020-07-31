#include "GPUSpikingModel.h"

#ifdef PERFORMANCE_METRICS
float g_time;
cudaEvent_t start, stop;
#endif // PERFORMANCE_METRICS

__constant__ int d_debug_mask[1];

GPUSpikingModel::GPUSpikingModel(Connections *conns, IAllNeurons *neurons, IAllSynapses *synapses, Layout *layout) : 	
  Model::Model(conns, neurons, synapses, layout),
  synapseIndexMapDevice(NULL),
  randNoise_d(NULL),
  allNeuronsDevice_(NULL),
  allSynapsesDevice_(NULL)
{
}

GPUSpikingModel::~GPUSpikingModel() 
{
  //Let Model base class handle de-allocation
}

/// Allocates  and initializes memories on CUDA device.
/// @param[out] allNeuronsDevice          Memory location of the pointer to the neurons list on device memory.
/// @param[out] allSynapsesDevice         Memory location of the pointer to the synapses list on device memory.
void GPUSpikingModel::allocDeviceStruct(void** allNeuronsDevice, void** allSynapsesDevice)
{
  // Allocate Neurons and Synapses structs on GPU device memory
  neurons_->allocNeuronDeviceStruct(allNeuronsDevice);
  synapses_->allocSynapseDeviceStruct(allSynapsesDevice);

  // Allocate memory for random noise array
  int neuron_count = Simulator::getInstance.getTotalNeurons();
  BGSIZE randNoise_d_size = neuron_count * sizeof (float);	// size of random noise array
  HANDLE_ERROR( cudaMalloc ( ( void ** ) &randNoise_d, randNoise_d_size ) );

  // Copy host neuron and synapse arrays into GPU device
  neurons_->copyNeuronHostToDevice( *allNeuronsDevice);
  synapses_->copySynapseHostToDevice( *allSynapsesDevice);

  // allocate synapse inverse map in device memory
  allocSynapseImap( neuron_count );
}

/// Copies device memories to host memories and deallocates them.
/// @param[out] allNeuronsDevice          Memory location of the pointer to the neurons list on device memory.
/// @param[out] allSynapsesDevice         Memory location of the pointer to the synapses list on device memory.
void GPUSpikingModel::deleteDeviceStruct(void** allNeuronsDevice, void** allSynapsesDevice)
{
  // copy device synapse and neuron structs to host memory
  neurons_->copyNeuronDeviceToHost( *allNeuronsDevice);
  // Deallocate device memory
  neurons_->deleteNeuronDeviceStruct( *allNeuronsDevice);
  // copy device synapse and neuron structs to host memory
  synapses_->copySynapseDeviceToHost( *allSynapsesDevice);
  // Deallocate device memory
  synapses_->deleteSynapseDeviceStruct( *allSynapsesDevice );
  deleteSynapseImap();
  HANDLE_ERROR( cudaFree( randNoise_d ) );
}

/// Sets up the Simulation.
void GPUSpikingModel::setupSim()
{
  // Set device ID
  HANDLE_ERROR( cudaSetDevice( g_deviceId ) );
  // Set DEBUG flag
  HANDLE_ERROR( cudaMemcpyToSymbol (d_debug_mask, &g_debug_mask, sizeof(int) ) );
  Model::setupSim();

  //initialize Mersenne Twister
  //assuming neuron_count >= 100 and is a multiple of 100. Note rng_mt_rng_count must be <= MT_RNG_COUNT
  int rng_blocks = 25; //# of blocks the kernel will use
  int rng_nPerRng = 4; //# of iterations per thread (thread granularity, # of rands generated per thread)
  int rng_mt_rng_count = Simulator::getInstance.getTotalNeurons()/rng_nPerRng; //# of threads to generate for neuron_count rand #s
  int rng_threads = rng_mt_rng_count/rng_blocks; //# threads per block needed
  initMTGPU(Simulator::getInstance().seed, rng_blocks, rng_threads, rng_nPerRng, rng_mt_rng_count);

#ifdef PERFORMANCE_METRICS
  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  t_gpu_rndGeneration = 0.0;
  t_gpu_advanceNeurons = 0.0;
  t_gpu_advanceSynapses = 0.0;
  t_gpu_calcSummation = 0.0;
#endif // PERFORMANCE_METRICS

  // allocates memories on CUDA device
  allocDeviceStruct((void **)&allNeuronsDevice_, (void **)&allSynapsesDevice_);

  // copy inverse map to the device memory
  copySynapseIndexMapHostToDevice(*synapseIndexMap_, Simulator::getInstance.getTotalNeurons());

  // set some parameters used for advanceNeuronsDevice
  neurons_->setAdvanceNeuronsDeviceParams(*synapses_);

  // set some parameters used for advanceSynapsesDevice
  synapses_->setAdvanceSynapsesDeviceParams();
}

/// Performs any finalization tasks on network following a simulation.
void GPUSpikingModel::cleanupSim()
{
  // deallocates memories on CUDA device
  deleteDeviceStruct((void**)&allNeuronsDevice_, (void**)&allSynapsesDevice_);

#ifdef PERFORMANCE_METRICS
  cudaEventDestroy( start );
  cudaEventDestroy( stop );
#endif // PERFORMANCE_METRICS
}

/// Advance everything in the model one time step. In this case, that
/// means calling all of the kernels that do the "micro step" updating
/// (i.e., NOT the stuff associated with growth).
void GPUSpikingModel::advance()
{
#ifdef PERFORMANCE_METRICS
  // Reset CUDA timer to start measurement of GPU operations
  cudaStartTimer();
#endif // PERFORMANCE_METRICS

  normalMTGPU(randNoise_d);

#ifdef PERFORMANCE_METRICS
  cudaLapTime(t_gpu_rndGeneration);
  cudaStartTimer();
#endif // PERFORMANCE_METRICS

  // display running info to console
  // Advance neurons ------------->
  neurons_->advanceNeurons(*synapses_, allNeuronsDevice_, allSynapsesDevice_, randNoise_d, synapseIndexMapDevice);

#ifdef PERFORMANCE_METRICS
  cudaLapTime(t_gpu_advanceNeurons);
  cudaStartTimer();
#endif // PERFORMANCE_METRICS

  // Advance synapses ------------->
  synapses_->advanceSynapses(allSynapsesDevice_, allNeuronsDevice_, synapseIndexMapDevice);

#ifdef PERFORMANCE_METRICS
  cudaLapTime(t_gpu_advanceSynapses);
  cudaStartTimer();
#endif // PERFORMANCE_METRICS

  // calculate summation point
  calcSummationMap();

#ifdef PERFORMANCE_METRICS
 cudaLapTime(t_gpu_calcSummation);
#endif // PERFORMANCE_METRICS
}

/// Add psr of all incoming synapses to summation points.
void GPUSpikingModel::calcSummationMap()
{
  // CUDA parameters
  const int threadsPerBlock = 256;
  int blocksPerGrid = ( Simulator::getInstance.getTotalNeurons() + threadsPerBlock - 1 ) / threadsPerBlock;

  calcSummationMapDevice <<< blocksPerGrid, threadsPerBlock >>> ( 
        Simulator::getInstance.getTotalNeurons(), allNeuronsDevice_, synapseIndexMapDevice, allSynapsesDevice_ );
}

/// Update the connection of all the Neurons and Synapses of the simulation.
void GPUSpikingModel::updateConnections()
{
  dynamic_cast<AllSpikingNeurons*>(neurons_)->copyNeuronDeviceSpikeCountsToHost(allNeuronsDevice_);
  dynamic_cast<AllSpikingNeurons*>(neurons_)->copyNeuronDeviceSpikeHistoryToHost(allNeuronsDevice_);

  // Update Connections data
  if (conns_->updateConnections(*neurons_, layout_)) {
    conns_->updateSynapsesWeights(Simulator::getInstance.getTotalNeurons(), *neurons_, *synapses_, allNeuronsDevice_, allSynapsesDevice_, layout_);
    // create synapse inverse map
    synapses_->createSynapseImap(synapseIndexMap_);
    // copy inverse map to the device memory
    copySynapseIndexMapHostToDevice(*synapseIndexMap_, Simulator::getInstance.getTotalNeurons());
  }
}

/// Update the Neuron's history.
void GPUSpikingModel::updateHistory()
{
  Model::updateHistory();
  // clear spike count
  dynamic_cast<AllSpikingNeurons*>(neurons_)->clearNeuronSpikeCounts(allNeuronsDevice_);
}

/// Allocate device memory for synapse inverse map.
/// @param  count	The number of neurons.
void GPUSpikingModel::allocSynapseImap( int count )
{
  SynapseIndexMap synapseIndexMap;

  HANDLE_ERROR( cudaMalloc( ( void ** ) &synapseIndexMap.outgoingSynapseBegin, count * sizeof( BGSIZE ) ) );
  HANDLE_ERROR( cudaMalloc( ( void ** ) &synapseIndexMap.outgoingSynapseCount, count * sizeof( BGSIZE ) ) );
  HANDLE_ERROR( cudaMemset(synapseIndexMap.outgoingSynapseBegin, 0, count * sizeof( BGSIZE ) ) );
  HANDLE_ERROR( cudaMemset(synapseIndexMap.outgoingSynapseCount, 0, count * sizeof( BGSIZE ) ) );

  HANDLE_ERROR( cudaMalloc( ( void ** ) &synapseIndexMap.incomingSynapseBegin, count * sizeof( BGSIZE ) ) );
  HANDLE_ERROR( cudaMalloc( ( void ** ) &synapseIndexMap.incomingSynapseCount, count * sizeof( BGSIZE ) ) );
  HANDLE_ERROR( cudaMemset(synapseIndexMap.incomingSynapseBegin, 0, count * sizeof( BGSIZE ) ) );
  HANDLE_ERROR( cudaMemset(synapseIndexMap.incomingSynapseCount, 0, count * sizeof( BGSIZE ) ) );

  HANDLE_ERROR( cudaMalloc( ( void ** ) &synapseIndexMapDevice, sizeof( SynapseIndexMap ) ) );
  HANDLE_ERROR( cudaMemcpy( synapseIndexMapDevice, &synapseIndexMap, sizeof( SynapseIndexMap ), 
        cudaMemcpyHostToDevice ) );
}

/// Deallocate device memory for synapse inverse map.
void GPUSpikingModel::deleteSynapseImap(  )
{
  SynapseIndexMap synapseIndexMap;

  HANDLE_ERROR( cudaMemcpy ( &synapseIndexMap, synapseIndexMapDevice, 
        sizeof( SynapseIndexMap ), cudaMemcpyDeviceToHost ) );

  HANDLE_ERROR( cudaFree( synapseIndexMap.outgoingSynapseBegin ) );
  HANDLE_ERROR( cudaFree( synapseIndexMap.outgoingSynapseCount ) );
  HANDLE_ERROR( cudaFree( synapseIndexMap.outgoingSynapseIndexMap ) );

  HANDLE_ERROR( cudaFree( synapseIndexMap.incomingSynapseBegin ) );
  HANDLE_ERROR( cudaFree( synapseIndexMap.incomingSynapseCount ) );
  HANDLE_ERROR( cudaFree( synapseIndexMap.incomingSynapseIndexMap ) );

  HANDLE_ERROR( cudaFree( synapseIndexMapDevice ) );
}

/// Copy SynapseIndexMap in host memory to SynapseIndexMap in device memory.
/// @param  synapseIndexMapHost		Reference to the SynapseIndexMap in host memory.
void GPUSpikingModel::copySynapseIndexMapHostToDevice(SynapseIndexMap &synapseIndexMapHost, int neuron_count)
{
  int total_synapse_counts = dynamic_cast<AllSynapses*>(synapses_)->total_synapse_counts;

  if (total_synapse_counts == 0)
    return;

  SynapseIndexMap synapseIndexMap;

  HANDLE_ERROR( cudaMemcpy ( &synapseIndexMap, synapseIndexMapDevice, 
        sizeof( SynapseIndexMap ), cudaMemcpyDeviceToHost ) );

  // forward map
  HANDLE_ERROR( cudaMemcpy ( synapseIndexMap.outgoingSynapseBegin, 
        synapseIndexMapHost.outgoingSynapseBegin, neuron_count * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy ( synapseIndexMap.outgoingSynapseCount, 
        synapseIndexMapHost.outgoingSynapseCount, neuron_count * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );
  // the number of synapses may change, so we reallocate the memory
  if (synapseIndexMap.outgoingSynapseIndexMap != NULL) {
    HANDLE_ERROR( cudaFree( synapseIndexMap.outgoingSynapseIndexMap ) );
  }
  HANDLE_ERROR( cudaMalloc( ( void ** ) &synapseIndexMap.outgoingSynapseIndexMap, 
        total_synapse_counts * sizeof( BGSIZE ) ) );
  HANDLE_ERROR( cudaMemcpy ( synapseIndexMap.outgoingSynapseIndexMap,synapseIndexMapHost.outgoingSynapseIndexMap, 
        total_synapse_counts * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );

  // active synapse map
  HANDLE_ERROR( cudaMemcpy ( synapseIndexMap.incomingSynapseBegin, 
        synapseIndexMapHost.incomingSynapseBegin, neuron_count * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy ( synapseIndexMap.incomingSynapseCount, 
        synapseIndexMapHost.incomingSynapseCount, neuron_count * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );
  // the number of synapses may change, so we reallocate the memory
  if (synapseIndexMap.incomingSynapseIndexMap != NULL) {
    HANDLE_ERROR( cudaFree( synapseIndexMap.incomingSynapseIndexMap ) );
  }
  HANDLE_ERROR( cudaMalloc( ( void ** ) &synapseIndexMap.incomingSynapseIndexMap, 
        total_synapse_counts * sizeof( BGSIZE ) ) );
  HANDLE_ERROR( cudaMemcpy ( synapseIndexMap.incomingSynapseIndexMap,synapseIndexMapHost.incomingSynapseIndexMap, 
        total_synapse_counts * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );

  HANDLE_ERROR( cudaMemcpy ( synapseIndexMapDevice, &synapseIndexMap, 
        sizeof( SynapseIndexMap ), cudaMemcpyHostToDevice ) );
}

/**
 * Calculate the sum of synaptic input to each neuron.
 *
 * Calculate the sum of synaptic input to each neuron. One thread
 * corresponds to one neuron. Iterates sequentially through the
 * forward synapse index map (synapseIndexMapDevice) to access only
 * existing synapses. Using this structure eliminates the need to skip
 * synapses that have undergone lazy deletion from the main
 * (allSynapsesDevice) synapse structure. The forward map is
 * re-computed during each network restructure (once per epoch) to
 * ensure that all synapse pointers for a neuron are stored
 * contiguously.
 * 
 * @param[in] totalNeurons           Number of neurons in the entire simulation.
 * @param[in,out] allNeuronsDevice   Pointer to Neuron structures in device memory.
 * @param[in] synapseIndexMapDevice  Pointer to forward map structures in device memory.
 * @param[in] allSynapsesDevice      Pointer to Synapse structures in device memory.
 */
__global__ void calcSummationMapDevice(int totalNeurons, 
				       AllSpikingNeuronsDeviceProperties* __restrict__ allNeuronsDevice, 
				       const SynapseIndexMap* __restrict__ synapseIndexMapDevice, 
				       const AllSpikingSynapsesDeviceProperties* __restrict__ allSynapsesDevice)
{
  // The usual thread ID calculation and guard against excess threads
  // (beyond the number of neurons, in this case).
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ( idx >= totalNeurons )
    return;

  // Number of incoming synapses
  const BGSIZE synCount = synapseIndexMapDevice->incomingSynapseCount[idx];
  // Optimization: terminate thread if no incoming synapses
  if (synCount != 0) {
    // Index of start of this neuron's block of forward map entries
    const int beginIndex = synapseIndexMapDevice->incomingSynapseBegin[idx];
    // Address of the start of this neuron's block of forward map entries
    const BGSIZE* activeMap_begin = 
      &(synapseIndexMapDevice->incomingSynapseIndexMap[beginIndex]);
    // Summed post-synaptic response (PSR)
    BGFLOAT sum = 0.0;
    // Index of the current incoming synapse
    BGSIZE synIndex;
    // Repeat for each incoming synapse
    for (BGSIZE i = 0; i < synCount; i++) {
      // Get index of current incoming synapse
      synIndex = activeMap_begin[i];
      // Fetch its PSR and add into sum
      sum += allSynapsesDevice->psr[synIndex];
    }
    // Store summed PSR into this neuron's summation point
    allNeuronsDevice->summation_map[idx] = sum;
  }
}

/// Copy GPU Synapse data to CPU.
void GPUSpikingModel::copyGPUtoCPU()
{
  // copy device synapse structs to host memory
  synapses_->copySynapseDeviceToHost(allSynapsesDevice_);
}

/// Copy CPU Synapse data to GPU.
void GPUSpikingModel::copyCPUtoGPU()
{
  // copy host synapse structs to device memory
  synapses_->copySynapseHostToDevice(allSynapsesDevice_);
}

/// Print out SynapseProps on the GPU.
void GPUSpikingModel::printGPUSynapsesPropsModel() const
{  
  synapses_->printGPUSynapsesProps(allSynapsesDevice_);
}

