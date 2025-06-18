/**
 * @file AllIFNeurons_d.cpp
 *
 * @brief A container of all Integate and Fire (IF) neuron data
 *
 * @ingroup Simulator/Vertices
 */

#include "AllIFNeurons.h"
#include "Book.h"
#include "GPUModel.h"
#include "Simulator.h"
#include "DeviceVector.h"

///  Allocate GPU memories to store all neurons' states,
///  and copy them from host to GPU memory.
///
///  @param  allVerticesDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
void AllIFNeurons::allocVerticesDeviceStruct()
{
   AllIFNeuronsDeviceProperties allNeurons;
   GPUModel *gpuModel = static_cast<GPUModel *>(&Simulator::getInstance().getModel());
   void **allVerticesDevice = reinterpret_cast<void **>(&(gpuModel->getAllVerticesDevice()));
   allocDeviceStruct(allNeurons);
   HANDLE_ERROR(cudaMalloc(allVerticesDevice, sizeof(AllIFNeuronsDeviceProperties)));
   HANDLE_ERROR(cudaMemcpy(*allVerticesDevice, &allNeurons, sizeof(AllIFNeuronsDeviceProperties),
                           cudaMemcpyHostToDevice));
}

///  Allocate GPU memories to store all neurons' states.
///  (Helper function of allocVerticesDeviceStruct)
///
///  @param  allVerticesDevice         GPU address of the AllIFNeuronsDeviceProperties struct.
void AllIFNeurons::allocDeviceStruct(AllIFNeuronsDeviceProperties &allVerticesDevice)
{
   int count = Simulator::getInstance().getTotalVertices();
   int maxSpikes = static_cast<int>(Simulator::getInstance().getEpochDuration()
                                    * Simulator::getInstance().getMaxFiringRate());

   C1_.allocateDeviceMemory();
   C2_.allocateDeviceMemory();
   Cm_.allocateDeviceMemory();
   I0_.allocateDeviceMemory();
   Iinject_.allocateDeviceMemory();
   Inoise_.allocateDeviceMemory();
   Isyn_.allocateDeviceMemory();
   Rm_.allocateDeviceMemory();
   Tau_.allocateDeviceMemory();
   Trefract_.allocateDeviceMemory();
   Vinit_.allocateDeviceMemory();
   Vm_.allocateDeviceMemory();
   Vrest_.allocateDeviceMemory();
   Vthresh_.allocateDeviceMemory();
   Vreset_.allocateDeviceMemory();
   numStepsInRefractoryPeriod_.allocateDeviceMemory();
   hasFired_.allocateDeviceMemory();
   summationPoints_.allocateDeviceMemory();

   //Handling memory operations for event buffer (device side) explicitly
   // since device vector does not support object types yet
#ifdef VALIDATION_MODE
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.spValidation_, count * sizeof(BGFLOAT)));
#endif
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.spikeHistory_, count * sizeof(uint64_t *)));

   uint64_t *pSpikeHistory[count];
   for (int i = 0; i < count; i++) {
      HANDLE_ERROR(cudaMalloc((void **)&pSpikeHistory[i], maxSpikes * sizeof(uint64_t)));
   }
   HANDLE_ERROR(cudaMemcpy(allVerticesDevice.spikeHistory_, pSpikeHistory,
                           count * sizeof(uint64_t *), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.bufferFront_, count * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.bufferEnd_, count * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.epochStart_, count * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.numElementsInEpoch_, count * sizeof(int)));
}

///  Delete GPU memories.
///
void AllIFNeurons::deleteVerticesDeviceStruct()
{
   AllIFNeuronsDeviceProperties allVerticesDeviceProps;
   GPUModel *gpuModel = static_cast<GPUModel *>(&Simulator::getInstance().getModel());
   void *allVerticesDevice = static_cast<void *>(gpuModel->getAllVerticesDevice());
   HANDLE_ERROR(cudaMemcpy(&allVerticesDeviceProps, allVerticesDevice,
                           sizeof(AllIFNeuronsDeviceProperties), cudaMemcpyDeviceToHost));
   deleteDeviceStruct(allVerticesDeviceProps);
   HANDLE_ERROR(cudaFree(allVerticesDevice));
}

///  Delete GPU memories.
///  (Helper function of deleteVerticesDeviceStruct)
///
///  @param  allVerticesDevice         GPU address of the AllIFNeuronsDeviceProperties struct.
void AllIFNeurons::deleteDeviceStruct(AllIFNeuronsDeviceProperties &allVerticesDevice)
{
   //Handling memory operations for event buffer (device side) explicitly
   // since device vector does not support object types yet
   int count = Simulator::getInstance().getTotalVertices();
   uint64_t *pSpikeHistory[count];
   HANDLE_ERROR(cudaMemcpy(pSpikeHistory, allVerticesDevice.spikeHistory_,
                           count * sizeof(uint64_t *), cudaMemcpyDeviceToHost));
   for (int i = 0; i < count; i++) {
      HANDLE_ERROR(cudaFree(pSpikeHistory[i]));
   }

   C1_.freeDeviceMemory();
   C2_.freeDeviceMemory();
   Cm_.freeDeviceMemory();
   I0_.freeDeviceMemory();
   Iinject_.freeDeviceMemory();
   Inoise_.freeDeviceMemory();
   Isyn_.freeDeviceMemory();
   Rm_.freeDeviceMemory();
   Tau_.freeDeviceMemory();
   Trefract_.freeDeviceMemory();
   Vinit_.freeDeviceMemory();
   Vm_.freeDeviceMemory();
   Vrest_.freeDeviceMemory();
   Vthresh_.freeDeviceMemory();
   Vreset_.freeDeviceMemory();
   numStepsInRefractoryPeriod_.freeDeviceMemory();
   hasFired_.freeDeviceMemory();
   summationPoints_.freeDeviceMemory();

#ifdef VALIDATION_MODE
   HANDLE_ERROR(cudaFree(allVerticesDevice.spValidation_));
#endif
   HANDLE_ERROR(cudaFree(allVerticesDevice.spikeHistory_));
}

///  Copy all neurons' data from host to device.
void AllIFNeurons::copyToDevice()
{
   C1_.copyToDevice();
   C2_.copyToDevice();
   Cm_.copyToDevice();
   I0_.copyToDevice();
   Iinject_.copyToDevice();
   Inoise_.copyToDevice();
   Isyn_.copyToDevice();
   Rm_.copyToDevice();
   Tau_.copyToDevice();
   Trefract_.copyToDevice();
   Vinit_.copyToDevice();
   Vm_.copyToDevice();
   Vrest_.copyToDevice();
   Vthresh_.copyToDevice();
   Vreset_.copyToDevice();
   numStepsInRefractoryPeriod_.copyToDevice();

   AllSpikingNeurons::copyToDevice();
}

///  Copy all neurons' data from device to host.
///
void AllIFNeurons::copyFromDevice()
{
   AllSpikingNeurons::copyFromDevice();

   C1_.copyToHost();
   C2_.copyToHost();
   Cm_.copyToHost();
   I0_.copyToHost();
   Iinject_.copyToHost();
   Inoise_.copyToHost();
   Isyn_.copyToHost();
   Rm_.copyToHost();
   Tau_.copyToHost();
   Trefract_.copyToHost();
   Vinit_.copyToHost();
   Vm_.copyToHost();
   Vrest_.copyToHost();
   Vthresh_.copyToHost();
   Vreset_.copyToHost();
   numStepsInRefractoryPeriod_.copyToHost();
}

///  Clear the spike counts out of all neurons.
///
///  @param  allVerticesDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
// TODO: Move this into EventBuffer somehow
void AllIFNeurons::clearVertexHistory(void *allVerticesDevice)
{
   AllIFNeuronsDeviceProperties allVerticesDeviceProps;
   HANDLE_ERROR(cudaMemcpy(&allVerticesDeviceProps, allVerticesDevice,
                           sizeof(AllIFNeuronsDeviceProperties), cudaMemcpyDeviceToHost));
   AllSpikingNeurons::clearDeviceSpikeCounts(allVerticesDeviceProps);
}


///  Update the state of all neurons for a time step
///  Notify outgoing synapses if neuron has fired.
///
///  @param  synapses               Reference to the allEdges struct on host memory.
///  @param  allVerticesDevice       GPU address of the AllIFNeuronsDeviceProperties struct
///                                 on device memory.
///  @param  allEdgesDevice      GPU address of the allEdgesDeviceProperties struct
///                                 on device memory.
///  @param  randNoise              Reference to the random noise array.
///  @param  edgeIndexMapDevice  GPU address of the EdgeIndexMap on device memory.
void AllIFNeurons::advanceVertices(AllEdges &synapses, void *allVerticesDevice,
                                   void *allEdgesDevice, float randNoise[],
                                   EdgeIndexMapDevice *edgeIndexMapDevice)
{
}