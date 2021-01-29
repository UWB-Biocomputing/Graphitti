/**
 * @file GPUSpikingModel.cu
 * 
 * @ingroup Simulation/Core
 *
 * @brief Implementation of Model for the spiking neural networks.
 * 
 */

#include "GPUSpikingModel.h"
#include "AllSynapsesDeviceFuncs.h"
#include "Connections.h"
#include "Global.h"
#include "IAllNeurons.h"
#include "IAllSynapses.h"

#ifdef PERFORMANCE_METRICS
float g_time;
cudaEvent_t start, stop;
#endif // PERFORMANCE_METRICS

__constant__ int d_debug_mask[1];

GPUSpikingModel::GPUSpikingModel() :
  Model::Model(),
  synapseIndexMapDevice_(NULL),
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
  // Get neurons and synapses
  shared_ptr<IAllNeurons> neurons = layout_->getNeurons();
  shared_ptr<IAllSynapses> synapses = connections_->getSynapses();

  // Allocate Neurons and Synapses structs on GPU device memory
  neurons->allocNeuronDeviceStruct(allNeuronsDevice);
  synapses->allocSynapseDeviceStruct(allSynapsesDevice);

  // Allocate memory for random noise array
  int numNeurons = Simulator::getInstance().getTotalNeurons();
  BGSIZE randNoise_d_size = numNeurons * sizeof (float);	// size of random noise array
  HANDLE_ERROR( cudaMalloc ( ( void ** ) &randNoise_d, randNoise_d_size ) );

  // Copy host neuron and synapse arrays into GPU device
  neurons->copyNeuronHostToDevice( *allNeuronsDevice );
  synapses->copySynapseHostToDevice( *allSynapsesDevice );

  // Allocate synapse inverse map in device memory
  allocSynapseImap( numNeurons );
}

/// Copies device memories to host memories and deallocates them.
/// @param[out] allNeuronsDevice          Memory location of the pointer to the neurons list on device memory.
/// @param[out] allSynapsesDevice         Memory location of the pointer to the synapses list on device memory.
void GPUSpikingModel::deleteDeviceStruct(void** allNeuronsDevice, void** allSynapsesDevice)
{  
  // Get neurons and synapses
  shared_ptr<IAllNeurons> neurons = layout_->getNeurons();
  shared_ptr<IAllSynapses> synapses = connections_->getSynapses();

  // Copy device synapse and neuron structs to host memory
  neurons->copyNeuronDeviceToHost( *allNeuronsDevice);
  // Deallocate device memory
  neurons->deleteNeuronDeviceStruct( *allNeuronsDevice);
  // Copy device synapse and neuron structs to host memory
  synapses->copySynapseDeviceToHost( *allSynapsesDevice);
  // Deallocate device memory
  synapses->deleteSynapseDeviceStruct( *allSynapsesDevice );
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
  //assuming numNeurons >= 100 and is a multiple of 100. Note rng_mt_rng_count must be <= MT_RNG_COUNT
  int rng_blocks = 25; //# of blocks the kernel will use
  int rng_nPerRng = 4; //# of iterations per thread (thread granularity, # of rands generated per thread)
  int rng_mt_rng_count = Simulator::getInstance().getTotalNeurons() / rng_nPerRng; //# of threads to generate for numNeurons rand #s
  int rng_threads = rng_mt_rng_count/rng_blocks; //# threads per block needed
  initMTGPU(Simulator::getInstance().getSeed(), rng_blocks, rng_threads, rng_nPerRng, rng_mt_rng_count);

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
  copySynapseIndexMapHostToDevice(*(connections_->getSynapseIndexMap().get()), Simulator::getInstance().getTotalNeurons());

  // set some parameters used for advanceNeuronsDevice
  layout_->getNeurons()->setAdvanceNeuronsDeviceParams(*(connections_->getSynapses().get()));

  // set some parameters used for advanceSynapsesDevice
  connections_->getSynapses()->setAdvanceSynapsesDeviceParams();
}

/// Performs any finalization tasks on network following a simulation.
void GPUSpikingModel::finish()
{
  // deallocates memories on CUDA device
  deleteDeviceStruct((void**)&allNeuronsDevice_, (void**)&allSynapsesDevice_);
  deleteSynapseImap();

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

  // Get neurons and synapses
  shared_ptr<IAllNeurons> neurons = layout_->getNeurons();
  shared_ptr<IAllSynapses> synapses = connections_->getSynapses();

  normalMTGPU(randNoise_d);

#ifdef PERFORMANCE_METRICS
  cudaLapTime(t_gpu_rndGeneration);
  cudaStartTimer();
#endif // PERFORMANCE_METRICS

  // display running info to console
  // Advance neurons ------------->
   dynamic_cast<AllSpikingNeurons *>(neurons.get())->advanceNeurons(*(connections_->getSynapses().get()), allNeuronsDevice_, allSynapsesDevice_, randNoise_d, synapseIndexMapDevice_);

#ifdef PERFORMANCE_METRICS
  cudaLapTime(t_gpu_advanceNeurons);
  cudaStartTimer();
#endif // PERFORMANCE_METRICS

  // Advance synapses ------------->
  synapses->advanceSynapses(allSynapsesDevice_, allNeuronsDevice_, synapseIndexMapDevice_);

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
  int blocksPerGrid = ( Simulator::getInstance().getTotalNeurons() + threadsPerBlock - 1 ) / threadsPerBlock;

  calcSummationMapDevice <<< blocksPerGrid, threadsPerBlock >>> (
        Simulator::getInstance().getTotalNeurons(), allNeuronsDevice_, synapseIndexMapDevice_, allSynapsesDevice_ );
}

/// Update the connection of all the Neurons and Synapses of the simulation.
void GPUSpikingModel::updateConnections()
{
  // Get neurons and synapses
  shared_ptr<IAllNeurons> neurons = layout_->getNeurons();
  shared_ptr<IAllSynapses> synapses = connections_->getSynapses();

  dynamic_cast<AllSpikingNeurons*>(neurons.get())->copyNeuronDeviceSpikeCountsToHost(allNeuronsDevice_);
  dynamic_cast<AllSpikingNeurons*>(neurons.get())->copyNeuronDeviceSpikeHistoryToHost(allNeuronsDevice_);

  // Update Connections data
  if (connections_->updateConnections(*(neurons.get()), layout_.get())) {
    connections_->updateSynapsesWeights(Simulator::getInstance().getTotalNeurons(), *(neurons.get()), *(synapses.get()), allNeuronsDevice_, allSynapsesDevice_, layout_.get());
    // create synapse index map
    connections_->createSynapseIndexMap();
    // copy index map to the device memory
    copySynapseIndexMapHostToDevice(*(connections_->getSynapseIndexMap().get()), Simulator::getInstance().getTotalNeurons());
  }
}

/// Update the Neuron's history.
void GPUSpikingModel::updateHistory()
{
  Model::updateHistory();
  // clear spike count
  
  shared_ptr<IAllNeurons> neurons = layout_->getNeurons();
  dynamic_cast<AllSpikingNeurons*>(neurons.get())->clearNeuronSpikeCounts(allNeuronsDevice_);
}

/// Allocate device memory for synapse inverse map.
/// @param  count	The number of neurons.
void GPUSpikingModel::allocSynapseImap( int count )
{
  SynapseIndexMap synapseIMapDevice;

  HANDLE_ERROR( cudaMalloc( ( void ** ) &synapseIMapDevice.outgoingSynapseBegin_, count * sizeof( BGSIZE ) ) );
  HANDLE_ERROR( cudaMalloc( ( void ** ) &synapseIMapDevice.outgoingSynapseCount_, count * sizeof( BGSIZE ) ) );
  HANDLE_ERROR( cudaMemset(synapseIMapDevice.outgoingSynapseBegin_, 0, count * sizeof( BGSIZE ) ) );
  HANDLE_ERROR( cudaMemset(synapseIMapDevice.outgoingSynapseCount_, 0, count * sizeof( BGSIZE ) ) );

  HANDLE_ERROR( cudaMalloc( ( void ** ) &synapseIMapDevice.incomingSynapseBegin_, count * sizeof( BGSIZE ) ) );
  HANDLE_ERROR( cudaMalloc( ( void ** ) &synapseIMapDevice.incomingSynapseCount_, count * sizeof( BGSIZE ) ) );
  HANDLE_ERROR( cudaMemset(synapseIMapDevice.incomingSynapseBegin_, 0, count * sizeof( BGSIZE ) ) );
  HANDLE_ERROR( cudaMemset(synapseIMapDevice.incomingSynapseCount_, 0, count * sizeof( BGSIZE ) ) );

  HANDLE_ERROR( cudaMalloc( ( void ** ) &synapseIndexMapDevice_, sizeof( SynapseIndexMap ) ) );
  HANDLE_ERROR( cudaMemcpy( synapseIndexMapDevice_, &synapseIMapDevice, sizeof( SynapseIndexMap ), 
        cudaMemcpyHostToDevice ) );
}

/// Deallocate device memory for synapse inverse map.
void GPUSpikingModel::deleteSynapseImap(  )
{
  SynapseIndexMap synapseIMapDevice;

  HANDLE_ERROR( cudaMemcpy ( &synapseIMapDevice, synapseIndexMapDevice_, 
        sizeof( SynapseIndexMap ), cudaMemcpyDeviceToHost ) );

  HANDLE_ERROR( cudaFree( synapseIMapDevice.outgoingSynapseBegin_ ) );
  HANDLE_ERROR( cudaFree( synapseIMapDevice.outgoingSynapseCount_ ) );
  HANDLE_ERROR( cudaFree( synapseIMapDevice.outgoingSynapseIndexMap_ ) );

  HANDLE_ERROR( cudaFree( synapseIMapDevice.incomingSynapseBegin_ ) );
  HANDLE_ERROR( cudaFree( synapseIMapDevice.incomingSynapseCount_ ) );
  HANDLE_ERROR( cudaFree( synapseIMapDevice.incomingSynapseIndexMap_ ) );

  HANDLE_ERROR( cudaFree( synapseIndexMapDevice_ ) );
}

/// Copy SynapseIndexMap in host memory to SynapseIndexMap in device memory.
/// @param  synapseIndexMapHost		Reference to the SynapseIndexMap in host memory.
void GPUSpikingModel::copySynapseIndexMapHostToDevice(SynapseIndexMap &synapseIndexMapHost, int numNeurons)
{
  shared_ptr<IAllSynapses> synapses = connections_->getSynapses();
  int totalSynapseCount = dynamic_cast<AllSynapses*>(synapses.get())->totalSynapseCount_;

  if (totalSynapseCount == 0)
    return;

  // TODO: rename variable, DevicePointer
  SynapseIndexMap synapseIMapDevice;

  HANDLE_ERROR( cudaMemcpy ( &synapseIMapDevice, synapseIndexMapDevice_, 
        sizeof( SynapseIndexMap ), cudaMemcpyDeviceToHost ) );

  // forward map
  HANDLE_ERROR( cudaMemcpy ( synapseIMapDevice.outgoingSynapseBegin_, 
        synapseIndexMapHost.outgoingSynapseBegin_, numNeurons * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy ( synapseIMapDevice.outgoingSynapseCount_, 
        synapseIndexMapHost.outgoingSynapseCount_, numNeurons * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );
  // the number of synapses may change, so we reallocate the memory
  if (synapseIMapDevice.outgoingSynapseIndexMap_ != NULL) {
    HANDLE_ERROR( cudaFree( synapseIMapDevice.outgoingSynapseIndexMap_ ) );
  }
  HANDLE_ERROR( cudaMalloc( ( void ** ) &synapseIMapDevice.outgoingSynapseIndexMap_, 
        totalSynapseCount * sizeof( BGSIZE ) ) );
  HANDLE_ERROR( cudaMemcpy ( synapseIMapDevice.outgoingSynapseIndexMap_, synapseIndexMapHost.outgoingSynapseIndexMap_, 
        totalSynapseCount * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );

  // active synapse map
  HANDLE_ERROR( cudaMemcpy ( synapseIMapDevice.incomingSynapseBegin_, 
        synapseIndexMapHost.incomingSynapseBegin_, numNeurons * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy ( synapseIMapDevice.incomingSynapseCount_, 
        synapseIndexMapHost.incomingSynapseCount_, numNeurons * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );
  // the number of synapses may change, so we reallocate the memory
  if (synapseIMapDevice.incomingSynapseIndexMap_ != NULL) {
    HANDLE_ERROR( cudaFree( synapseIMapDevice.incomingSynapseIndexMap_ ) );
  }
  HANDLE_ERROR( cudaMalloc( ( void ** ) &synapseIMapDevice.incomingSynapseIndexMap_, 
        totalSynapseCount * sizeof( BGSIZE ) ) );
  HANDLE_ERROR( cudaMemcpy ( synapseIMapDevice.incomingSynapseIndexMap_, synapseIndexMapHost.incomingSynapseIndexMap_, 
        totalSynapseCount * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );

  HANDLE_ERROR( cudaMemcpy ( synapseIndexMapDevice_, &synapseIMapDevice, 
        sizeof( SynapseIndexMap ), cudaMemcpyHostToDevice ) );
}

/// Calculate the sum of synaptic input to each neuron.
///
/// Calculate the sum of synaptic input to each neuron. One thread
/// corresponds to one neuron. Iterates sequentially through the
/// forward synapse index map (synapseIndexMapDevice_) to access only
/// existing synapses. Using this structure eliminates the need to skip
/// synapses that have undergone lazy deletion from the main
/// (allSynapsesDevice) synapse structure. The forward map is
/// re-computed during each network restructure (once per epoch) to
/// ensure that all synapse pointers for a neuron are stored
/// contiguously.
/// 
/// @param[in] totalNeurons           Number of neurons in the entire simulation.
/// @param[in,out] allNeuronsDevice   Pointer to Neuron structures in device memory.
/// @param[in] synapseIndexMapDevice_  Pointer to forward map structures in device memory.
/// @param[in] allSynapsesDevice      Pointer to Synapse structures in device memory.
__global__ void calcSummationMapDevice(int totalNeurons, 
				       AllSpikingNeuronsDeviceProperties* __restrict__ allNeuronsDevice, 
				       const SynapseIndexMap* __restrict__ synapseIndexMapDevice_, 
				       const AllSpikingSynapsesDeviceProperties* __restrict__ allSynapsesDevice)
{
  // The usual thread ID calculation and guard against excess threads
  // (beyond the number of neurons, in this case).
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ( idx >= totalNeurons )
    return;

  // Number of incoming synapses
  const BGSIZE synCount = synapseIndexMapDevice_->incomingSynapseCount_[idx];
  // Optimization: terminate thread if no incoming synapses
  if (synCount != 0) {
    // Index of start of this neuron's block of forward map entries
    const int beginIndex = synapseIndexMapDevice_->incomingSynapseBegin_[idx];
    // Address of the start of this neuron's block of forward map entries
    const BGSIZE* activeMapBegin = 
      &(synapseIndexMapDevice_->incomingSynapseIndexMap_[beginIndex]);
    // Summed post-synaptic response (PSR)
    BGFLOAT sum = 0.0;
    // Index of the current incoming synapse
    BGSIZE synIndex;
    // Repeat for each incoming synapse
    for (BGSIZE i = 0; i < synCount; i++) {
      // Get index of current incoming synapse
      synIndex = activeMapBegin[i];
      // Fetch its PSR and add into sum
      sum += allSynapsesDevice->psr_[synIndex];
    }
    // Store summed PSR into this neuron's summation point
    allNeuronsDevice->summationMap_[idx] = sum;
  }
}

/// Copy GPU Synapse data to CPU.
void GPUSpikingModel::copyGPUtoCPU()
{
  // copy device synapse structs to host memory
  connections_->getSynapses()->copySynapseDeviceToHost(allSynapsesDevice_);
}

/// Copy CPU Synapse data to GPU.
void GPUSpikingModel::copyCPUtoGPU()
{
  // copy host synapse structs to device memory
  connections_->getSynapses()->copySynapseHostToDevice(allSynapsesDevice_);
}

/// Print out SynapseProps on the GPU.
void GPUSpikingModel::printGPUSynapsesPropsModel() const
{  
  connections_->getSynapses()->printGPUSynapsesProps(allSynapsesDevice_);
}

