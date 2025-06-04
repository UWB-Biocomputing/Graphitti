/**
 * @file GPUModel.cpp
 * 
 * @ingroup Simulator/Core
 *
 * @brief Implementation of Model for the graph-based networks.
 * 
 */

#include "GPUModel.h"
#include "AllEdges.h"
#include "AllVertices.h"
#include "Connections.h"
#include "Global.h"
#include "OperationManager.h"

#ifdef VALIDATION_MODE
   #include "AllIFNeurons.h"
   #include "OperationManager.h"
#endif
#ifdef PERFORMANCE_METRICS
float g_time;
cudaEvent_t start, stop;
#endif   // PERFORMANCE_METRICS

__constant__ int d_debug_mask[1];

GPUModel::GPUModel() :
   Model::Model(), edgeIndexMapDevice_(nullptr), randNoise_d(nullptr), allVerticesDevice_(nullptr),
   allEdgesDevice_(nullptr)
{
   // Register allocNeuronDeviceStruct function as a allocateGPU operation in the OperationManager
   function<void()> allocateGPU = bind(&GPUModel::allocDeviceStruct, this);
   OperationManager::getInstance().registerOperation(Operations::allocateGPU, allocateGPU);

   // Register copyCPUtoGPU function as a copyCPUtoGPU operation in the OperationManager
   function<void()> copyCPUtoGPU = bind(&GPUModel::copyCPUtoGPU, this);
   OperationManager::getInstance().registerOperation(Operations::copyToGPU, copyCPUtoGPU);

   // Note: We do not register a corresponding copyFromGPU operation here because
   // we are only copying the synapseIndexMap to the GPU. This map is a read-only lookup table
   // that gets recreated from scratch on each update. As a result, we only need to allocate,
   // copy to GPU, and deallocate — there is no meaningful data to copy back from the GPU.

   // Register deleteSynapseImap function as a deallocateGPUMemory operation in the OperationManager
   function<void()> deallocateGPUMemory = bind(&GPUModel::deleteDeviceStruct, this);
   OperationManager::getInstance().registerOperation(Operations::deallocateGPUMemory,
                                                     deallocateGPUMemory);
}

/// Allocates  and initializes memories on CUDA device.
void GPUModel::allocDeviceStruct()
{
   // Allocate memory for random noise array
   int numVertices = Simulator::getInstance().getTotalVertices();
   BGSIZE randNoise_d_size = numVertices * sizeof(float);   // size of random noise array
   HANDLE_ERROR(cudaMalloc((void **)&randNoise_d, randNoise_d_size));

   // Allocate synapse inverse map in device memory
   allocEdgeIndexMap(numVertices);
}

/// Copies device memories to host memories and deallocates them.
void GPUModel::deleteDeviceStruct()
{
   // Deallocate device memory
   EdgeIndexMapDevice synapseIMapDevice;
   HANDLE_ERROR(cudaMemcpy(&synapseIMapDevice, edgeIndexMapDevice_, sizeof(EdgeIndexMapDevice),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaFree(synapseIMapDevice.outgoingEdgeBegin_));
   HANDLE_ERROR(cudaFree(synapseIMapDevice.outgoingEdgeCount_));
   HANDLE_ERROR(cudaFree(synapseIMapDevice.outgoingEdgeIndexMap_));
   HANDLE_ERROR(cudaFree(synapseIMapDevice.incomingEdgeBegin_));
   HANDLE_ERROR(cudaFree(synapseIMapDevice.incomingEdgeCount_));
   HANDLE_ERROR(cudaFree(synapseIMapDevice.incomingEdgeIndexMap_));
   HANDLE_ERROR(cudaFree(edgeIndexMapDevice_));
   HANDLE_ERROR(cudaFree(randNoise_d));
}

/// Sets up the Simulation.
void GPUModel::setupSim()
{
   // Set device ID
   HANDLE_ERROR(cudaSetDevice(g_deviceId));
   // Set DEBUG flag
   HANDLE_ERROR(cudaMemcpyToSymbol(d_debug_mask, &g_debug_mask, sizeof(int)));
   Model::setupSim();

   //initialize Mersenne Twister
   //assuming numVertices >= 100 and is a multiple of 100. Note rng_mt_rng_count must be <= MT_RNG_COUNT
   int rng_blocks = 25;   //# of blocks the kernel will use
   int rng_nPerRng
      = 4;   //# of iterations per thread (thread granularity, # of rands generated per thread)
   int rng_mt_rng_count = Simulator::getInstance().getTotalVertices()
                          / rng_nPerRng;   //# of threads to generate for numVertices rand #s
   int rng_threads = rng_mt_rng_count / rng_blocks;   //# threads per block needed
   initMTGPU(Simulator::getInstance().getNoiseRngSeed(), rng_blocks, rng_threads, rng_nPerRng,
             rng_mt_rng_count);

#ifdef PERFORMANCE_METRICS
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   t_gpu_rndGeneration = 0.0;
   t_gpu_advanceNeurons = 0.0;
   t_gpu_advanceSynapses = 0.0;
   t_gpu_calcSummation = 0.0;
#endif   // PERFORMANCE_METRICS
   // Allocate and copy neuron/synapse data structures to GPU memory
   OperationManager::getInstance().executeOperation(Operations::allocateGPU);
   OperationManager::getInstance().executeOperation(Operations::copyToGPU);

   AllEdges &edges = connections_->getEdges();
   // set some parameters used for advanceVerticesDevice
   layout_->getVertices().setAdvanceVerticesDeviceParams(edges);

   // set some parameters used for advanceEdgesDevice
   edges.setAdvanceEdgesDeviceParams();
}

/// Performs any finalization tasks on network following a simulation.
void GPUModel::finish()
{
   // copy device synapse and neuron structs to host memory
   OperationManager::getInstance().executeOperation(Operations::copyFromGPU);
   // deallocates memories on CUDA device
   OperationManager::getInstance().executeOperation(Operations::deallocateGPUMemory);

#ifdef PERFORMANCE_METRICS
   cudaEventDestroy(start);
   cudaEventDestroy(stop);
#endif   // PERFORMANCE_METRICS
}

/// Advance everything in the model one time step. In this case, that
/// means calling all of the kernels that do the "micro step" updating
/// (i.e., NOT the stuff associated with growth).
void GPUModel::advance()
{
#ifdef PERFORMANCE_METRICS
   // Reset CUDA timer to start measurement of GPU operations
   cudaStartTimer();
#endif   // PERFORMANCE_METRICS

   // Get vertices and edges
   AllVertices &vertices = layout_->getVertices();
   AllEdges &edges = connections_->getEdges();

#ifdef VALIDATION_MODE
   int verts = Simulator::getInstance().getTotalVertices();
   std::vector<float> randNoise_h(verts);
   for (int i = verts - 1; i >= 0; i--) {
      randNoise_h[i] = (*noiseRNG)();
   }
   //static int testNumbers = 0;
   // for (int i = 0; i < verts; i++) {
   //    outFile << "index: " << i << " " << randNoise_h[i] << endl;
   // }
   cudaMemcpy(randNoise_d, randNoise_h.data(), verts * sizeof(float), cudaMemcpyHostToDevice);
#else
   normalMTGPU(randNoise_d);
#endif
//LOG4CPLUS_DEBUG(vertexLogger_, "Index: " << index << " Vm: " << Vm);
#ifdef PERFORMANCE_METRICS
   cudaLapTime(t_gpu_rndGeneration);
   cudaStartTimer();
#endif   // PERFORMANCE_METRICS

   // display running info to console
   // Advance vertices ------------->
   vertices.advanceVertices(edges, allVerticesDevice_, allEdgesDevice_, randNoise_d,
                            edgeIndexMapDevice_);
#ifdef VALIDATION_MODE
   //(AllIFNeuronsDeviceProperties *)allVerticesDevice,
   log4cplus::Logger vertexLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("vertex"));
   std::vector<float> sp_h(verts);
   std::vector<float> vm_h(verts);
   std::vector<float> Inoise_h(verts);
   AllIFNeuronsDeviceProperties validationNeurons;
   HANDLE_ERROR(cudaMemcpy((void *)&validationNeurons, allVerticesDevice_,
                           sizeof(AllIFNeuronsDeviceProperties), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(sp_h.data(), validationNeurons.spValidation_, verts * sizeof(float),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(vm_h.data(), validationNeurons.Vm_, verts * sizeof(float),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(Inoise_h.data(), validationNeurons.Inoise_, verts * sizeof(float),
                           cudaMemcpyDeviceToHost));

   for (int i = verts - 1; i >= 0; i--) {
      LOG4CPLUS_DEBUG(vertexLogger_, endl
                                        << "Advance Index[" << i << "] :: Noise = "
                                        << randNoise_h[i] << "\tVm: " << vm_h[i] << endl
                                        << "\tsp = " << sp_h[i] << endl
                                        << "\tInoise = " << Inoise_h[i] << endl);
   }
#endif
//LOG4CPLUS_DEBUG(vertexLogger_, "ADVANCE NEURON LIF[" << index << "] :: Noise = " << noise);
//LOG4CPLUS_DEBUG(vertexLogger_, "Index: " << index << " Vm: " << Vm);
// LOG4CPLUS_DEBUG(vertexLogger_, "NEURON[" << index << "] {" << endl
//                                          << "\tVm = " << Vm << endl
//                                          << "\tVthresh = " << Vthresh << endl
//                                          << "\tsummationPoint = " << summationPoint << endl
//                                          << "\tI0 = " << I0 << endl
//                                          << "\tInoise = " << Inoise << endl
//                                          << "\tC1 = " << C1 << endl
//                                          << "\tC2 = " << C2 << endl
//                                          << "}" << endl);
#ifdef PERFORMANCE_METRICS
   cudaLapTime(t_gpu_advanceNeurons);
   cudaStartTimer();
#endif   // PERFORMANCE_METRICS

   // Advance edges ------------->
   edges.advanceEdges(allEdgesDevice_, allVerticesDevice_, edgeIndexMapDevice_);

#ifdef PERFORMANCE_METRICS
   cudaLapTime(t_gpu_advanceSynapses);
   cudaStartTimer();
#endif   // PERFORMANCE_METRICS

   // integrate the inputs of the vertices
   vertices.integrateVertexInputs(allVerticesDevice_, edgeIndexMapDevice_, allEdgesDevice_);

#ifdef PERFORMANCE_METRICS
   cudaLapTime(t_gpu_calcSummation);
#endif   // PERFORMANCE_METRICS
}

/// Update the connection of all the vertices and edges of the simulation.
void GPUModel::updateConnections()
{
   // Get vertices and edges
   AllVertices &vertices = layout_->getVertices();
   AllEdges &edges = connections_->getEdges();

   vertices.copyFromDevice();

   // Update Connections data
   if (connections_->updateConnections(vertices)) {
      connections_->updateEdgesWeights(Simulator::getInstance().getTotalVertices(), vertices, edges,
                                       allVerticesDevice_, allEdgesDevice_, getLayout());
      // create edge index map
      connections_->createEdgeIndexMap();
      // copy index map to the device memory
      copyCPUtoGPU();
   }
}

/// Update the vertex's history.
void GPUModel::updateHistory()
{
   Model::updateHistory();

   layout_->getVertices().clearVertexHistory(allVerticesDevice_);
}

/// Allocate device memory for edge inverse map.
/// @param  count	The number of vertices.
void GPUModel::allocEdgeIndexMap(int count)
{
   EdgeIndexMapDevice edgeIndexMapDevice;

   HANDLE_ERROR(
      cudaMalloc((void **)&edgeIndexMapDevice.outgoingEdgeBegin_, count * sizeof(BGSIZE)));
   HANDLE_ERROR(
      cudaMalloc((void **)&edgeIndexMapDevice.outgoingEdgeCount_, count * sizeof(BGSIZE)));
   HANDLE_ERROR(cudaMemset(edgeIndexMapDevice.outgoingEdgeBegin_, 0, count * sizeof(BGSIZE)));
   HANDLE_ERROR(cudaMemset(edgeIndexMapDevice.outgoingEdgeCount_, 0, count * sizeof(BGSIZE)));
   HANDLE_ERROR(
      cudaMalloc((void **)&edgeIndexMapDevice.incomingEdgeBegin_, count * sizeof(BGSIZE)));
   HANDLE_ERROR(
      cudaMalloc((void **)&edgeIndexMapDevice.incomingEdgeCount_, count * sizeof(BGSIZE)));
   HANDLE_ERROR(cudaMemset(edgeIndexMapDevice.incomingEdgeBegin_, 0, count * sizeof(BGSIZE)));
   HANDLE_ERROR(cudaMemset(edgeIndexMapDevice.incomingEdgeCount_, 0, count * sizeof(BGSIZE)));
   HANDLE_ERROR(cudaMalloc((void **)&edgeIndexMapDevice_, sizeof(EdgeIndexMapDevice)));
   edgeIndexMapDevice.incomingEdgeIndexMap_ = nullptr;
   edgeIndexMapDevice.outgoingEdgeIndexMap_ = nullptr;
   HANDLE_ERROR(cudaMemcpy(edgeIndexMapDevice_, &edgeIndexMapDevice, sizeof(EdgeIndexMapDevice),
                           cudaMemcpyHostToDevice));
}

/// Calculate the sum of synaptic input to each neuron.
///
/// Calculate the sum of synaptic input to each neuron. One thread
/// corresponds to one neuron. Iterates sequentially through the
/// forward synapse index map (edgeIndexMapDevice_) to access only
/// existing synapses. Using this structure eliminates the need to skip
/// synapses that have undergone lazy deletion from the main
/// (allEdgesDevice) synapse structure. The forward map is
/// re-computed during each network restructure (once per epoch) to
/// ensure that all synapse pointers for a neuron are stored
/// contiguously.
///
/// @param[in] totalVertices           Number of vertices in the entire simulation.
/// @param[in,out] allVerticesDevice   Pointer to Neuron structures in device memory.
/// @param[in] edgeIndexMapDevice_  Pointer to forward map structures in device memory.
/// @param[in] allEdgesDevice      Pointer to Synapse structures in device memory.
__global__ void
   calcSummationPointDevice(int totalVertices,
                            AllSpikingNeuronsDeviceProperties *__restrict__ allVerticesDevice,
                            const EdgeIndexMapDevice *__restrict__ edgeIndexMapDevice_,
                            const AllSpikingSynapsesDeviceProperties *__restrict__ allEdgesDevice)
{
   // The usual thread ID calculation and guard against excess threads
   // (beyond the number of vertices, in this case).
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx >= totalVertices)
      return;

   // Number of incoming synapses
   const BGSIZE synCount = edgeIndexMapDevice_->incomingEdgeCount_[idx];
   // Optimization: terminate thread if no incoming synapses
   if (synCount != 0) {
      // Index of start of this neuron's block of forward map entries
      const int beginIndex = edgeIndexMapDevice_->incomingEdgeBegin_[idx];
      // Address of the start of this neuron's block of forward map entries
      const BGSIZE *activeMapBegin = &(edgeIndexMapDevice_->incomingEdgeIndexMap_[beginIndex]);
      // Summed post-synaptic response (PSR)
      BGFLOAT sum = 0.0;
      // Index of the current incoming synapse
      BGSIZE synIndex;
      // Repeat for each incoming synapse
      for (BGSIZE i = 0; i < synCount; i++) {
         // Get index of current incoming synapse
         synIndex = activeMapBegin[i];
         // Fetch its PSR and add into sum
         sum += allEdgesDevice->psr_[synIndex];
      }
      // Store summed PSR into this neuron's summation point
      allVerticesDevice->summationPoints_[idx] = sum;
   }
}

/// Allocate and Copy CPU Synapse data to GPU.
void GPUModel::copyCPUtoGPU()
{
   EdgeIndexMap synapseIndexMapHost = connections_->getEdgeIndexMap();
   int numVertices = Simulator::getInstance().getTotalVertices();
   AllEdges &synapses = connections_->getEdges();
   int totalSynapseCount = dynamic_cast<AllEdges &>(synapses).totalEdgeCount_;
   if (totalSynapseCount == 0)
      return;
   EdgeIndexMapDevice synapseIMapDevice;
   HANDLE_ERROR(cudaMemcpy(&synapseIMapDevice, edgeIndexMapDevice_, sizeof(EdgeIndexMapDevice),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(synapseIMapDevice.outgoingEdgeBegin_,
                           synapseIndexMapHost.outgoingEdgeBegin_.data(),
                           numVertices * sizeof(BGSIZE), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(synapseIMapDevice.outgoingEdgeCount_,
                           synapseIndexMapHost.outgoingEdgeCount_.data(),
                           numVertices * sizeof(BGSIZE), cudaMemcpyHostToDevice));
   if (synapseIMapDevice.outgoingEdgeIndexMap_ != nullptr) {
      HANDLE_ERROR(cudaFree(synapseIMapDevice.outgoingEdgeIndexMap_));
   }
   HANDLE_ERROR(cudaMalloc((void **)&synapseIMapDevice.outgoingEdgeIndexMap_,
                           totalSynapseCount * sizeof(BGSIZE)));
   HANDLE_ERROR(cudaMemcpy(synapseIMapDevice.outgoingEdgeIndexMap_,
                           synapseIndexMapHost.outgoingEdgeIndexMap_.data(),
                           totalSynapseCount * sizeof(BGSIZE), cudaMemcpyHostToDevice));
   // active synapse map
   HANDLE_ERROR(cudaMemcpy(synapseIMapDevice.incomingEdgeBegin_,
                           synapseIndexMapHost.incomingEdgeBegin_.data(),
                           numVertices * sizeof(BGSIZE), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(synapseIMapDevice.incomingEdgeCount_,
                           synapseIndexMapHost.incomingEdgeCount_.data(),
                           numVertices * sizeof(BGSIZE), cudaMemcpyHostToDevice));
   // the number of synapses may change, so we reallocate the memory
   if (synapseIMapDevice.incomingEdgeIndexMap_ != nullptr) {
      HANDLE_ERROR(cudaFree(synapseIMapDevice.incomingEdgeIndexMap_));
      synapseIMapDevice.incomingEdgeIndexMap_ = nullptr;
   }
   HANDLE_ERROR(cudaMalloc((void **)&synapseIMapDevice.incomingEdgeIndexMap_,
                           totalSynapseCount * sizeof(BGSIZE)));
   HANDLE_ERROR(cudaMemcpy(synapseIMapDevice.incomingEdgeIndexMap_,
                           synapseIndexMapHost.incomingEdgeIndexMap_.data(),
                           totalSynapseCount * sizeof(BGSIZE), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(edgeIndexMapDevice_, &synapseIMapDevice, sizeof(EdgeIndexMapDevice),
                           cudaMemcpyHostToDevice));
}

/// Print out EdgeProps on the GPU.
void GPUModel::printGPUEdgesPropsModel() const
{
   connections_->getEdges().printGPUEdgesProps(allEdgesDevice_);
}

/// Getter for neuron structure in device memory
AllVerticesDeviceProperties *&GPUModel::getAllVerticesDevice()
{
   return allVerticesDevice_;
}

/// Getter for synapse structures in device memory
AllEdgesDeviceProperties *&GPUModel::getAllEdgesDevice()
{
   return allEdgesDevice_;
}