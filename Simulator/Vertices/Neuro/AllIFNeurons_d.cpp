/**
 * @file AllIFNeurons_d.cpp
 *
 * @brief A container of all Integate and Fire (IF) neuron data
 *
 * @ingroup Simulator/Vertices
 */

#include "AllIFNeurons.h"
#include "Book.h"

///  Allocate GPU memories to store all neurons' states,
///  and copy them from host to GPU memory.
///
///  @param  allVerticesDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
void AllIFNeurons::allocNeuronDeviceStruct(void **allVerticesDevice)
{
   AllIFNeuronsDeviceProperties allNeurons;
   allocDeviceStruct(allNeurons);
   HANDLE_ERROR(cudaMalloc(allVerticesDevice, sizeof(AllIFNeuronsDeviceProperties)));
   HANDLE_ERROR(cudaMemcpy(*allVerticesDevice, &allNeurons, sizeof(AllIFNeuronsDeviceProperties),
                           cudaMemcpyHostToDevice));
}

///  Allocate GPU memories to store all neurons' states.
///  (Helper function of allocNeuronDeviceStruct)
///
///  @param  allVerticesDevice         GPU address of the AllIFNeuronsDeviceProperties struct.
void AllIFNeurons::allocDeviceStruct(AllIFNeuronsDeviceProperties &allVerticesDevice)
{
   int count = Simulator::getInstance().getTotalVertices();
   int maxSpikes = static_cast<int>(Simulator::getInstance().getEpochDuration()
                                    * Simulator::getInstance().getMaxFiringRate());
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.C1_, count * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.C2_, count * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.Cm_, count * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.I0_, count * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.Iinject_, count * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.Inoise_, count * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.Isyn_, count * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.Rm_, count * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.Tau_, count * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.Trefract_, count * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.Vinit_, count * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.Vm_, count * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.Vreset_, count * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.Vrest_, count * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.Vthresh_, count * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.hasFired_, count * sizeof(bool)));
   HANDLE_ERROR(
      cudaMalloc((void **)&allVerticesDevice.numStepsInRefractoryPeriod_, count * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allVerticesDevice.summationPoints_, count * sizeof(BGFLOAT)));
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
///  @param  allVerticesDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
void AllIFNeurons::deleteNeuronDeviceStruct(void *allVerticesDevice)
{
   AllIFNeuronsDeviceProperties allVerticesDeviceProps;
   HANDLE_ERROR(cudaMemcpy(&allVerticesDeviceProps, allVerticesDevice,
                           sizeof(AllIFNeuronsDeviceProperties), cudaMemcpyDeviceToHost));
   deleteDeviceStruct(allVerticesDeviceProps);
   HANDLE_ERROR(cudaFree(allVerticesDevice));
}

///  Delete GPU memories.
///  (Helper function of deleteNeuronDeviceStruct)
///
///  @param  allVerticesDevice         GPU address of the AllIFNeuronsDeviceProperties struct.
void AllIFNeurons::deleteDeviceStruct(AllIFNeuronsDeviceProperties &allVerticesDevice)
{
   int count = Simulator::getInstance().getTotalVertices();
   uint64_t *pSpikeHistory[count];
   HANDLE_ERROR(cudaMemcpy(pSpikeHistory, allVerticesDevice.spikeHistory_,
                           count * sizeof(uint64_t *), cudaMemcpyDeviceToHost));
   for (int i = 0; i < count; i++) {
      HANDLE_ERROR(cudaFree(pSpikeHistory[i]));
   }
   HANDLE_ERROR(cudaFree(allVerticesDevice.C1_));
   HANDLE_ERROR(cudaFree(allVerticesDevice.C2_));
   HANDLE_ERROR(cudaFree(allVerticesDevice.Cm_));
   HANDLE_ERROR(cudaFree(allVerticesDevice.I0_));
   HANDLE_ERROR(cudaFree(allVerticesDevice.Iinject_));
   HANDLE_ERROR(cudaFree(allVerticesDevice.Inoise_));
   HANDLE_ERROR(cudaFree(allVerticesDevice.Isyn_));
   HANDLE_ERROR(cudaFree(allVerticesDevice.Rm_));
   HANDLE_ERROR(cudaFree(allVerticesDevice.Tau_));
   HANDLE_ERROR(cudaFree(allVerticesDevice.Trefract_));
   HANDLE_ERROR(cudaFree(allVerticesDevice.Vinit_));
   HANDLE_ERROR(cudaFree(allVerticesDevice.Vm_));
   HANDLE_ERROR(cudaFree(allVerticesDevice.Vreset_));
   HANDLE_ERROR(cudaFree(allVerticesDevice.Vrest_));
   HANDLE_ERROR(cudaFree(allVerticesDevice.Vthresh_));
   HANDLE_ERROR(cudaFree(allVerticesDevice.hasFired_));
   HANDLE_ERROR(cudaFree(allVerticesDevice.numStepsInRefractoryPeriod_));
   HANDLE_ERROR(cudaFree(allVerticesDevice.summationPoints_));
   HANDLE_ERROR(cudaFree(allVerticesDevice.spikeHistory_));
}

///  Copy all neurons' data from host to device.
///
///  @param  allVerticesDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
void AllIFNeurons::copyToDevice(void *allVerticesDevice)
{
   int count = Simulator::getInstance().getTotalVertices();
   AllIFNeuronsDeviceProperties allVerticesDeviceProps;
   HANDLE_ERROR(cudaMemcpy(&allVerticesDeviceProps, allVerticesDevice,
                           sizeof(AllIFNeuronsDeviceProperties), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(allVerticesDeviceProps.C1_, C1_.data(), count * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allVerticesDeviceProps.C2_, C2_.data(), count * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allVerticesDeviceProps.Cm_, Cm_.data(), count * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allVerticesDeviceProps.I0_, I0_.data(), count * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allVerticesDeviceProps.Iinject_, Iinject_.data(),
                           count * sizeof(BGFLOAT), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allVerticesDeviceProps.Inoise_, Inoise_.data(), count * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allVerticesDeviceProps.Isyn_, Isyn_.data(), count * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allVerticesDeviceProps.Rm_, Rm_.data(), count * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allVerticesDeviceProps.Tau_, Tau_.data(), count * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allVerticesDeviceProps.Trefract_, Trefract_.data(),
                           count * sizeof(BGFLOAT), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allVerticesDeviceProps.Vinit_, Vinit_.data(), count * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allVerticesDeviceProps.Vm_, Vm_.data(), count * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allVerticesDeviceProps.Vreset_, Vreset_.data(), count * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allVerticesDeviceProps.Vrest_, Vrest_.data(), count * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allVerticesDeviceProps.Vthresh_, Vthresh_.data(),
                           count * sizeof(BGFLOAT), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allVerticesDeviceProps.numStepsInRefractoryPeriod_,
                           numStepsInRefractoryPeriod_.data(), count * sizeof(int),
                           cudaMemcpyHostToDevice));
   AllSpikingNeurons::copyToDevice(allVerticesDevice);
}

///  Copy all neurons' data from device to host.
///
///  @param  allVerticesDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
void AllIFNeurons::copyFromDevice(void *allVerticesDevice)
{
   AllSpikingNeurons::copyFromDevice(allVerticesDevice);
   int count = Simulator::getInstance().getTotalVertices();
   AllIFNeuronsDeviceProperties allVerticesDeviceProps;
   HANDLE_ERROR(cudaMemcpy(&allVerticesDeviceProps, allVerticesDevice,
                           sizeof(AllIFNeuronsDeviceProperties), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(C1_.data(), allVerticesDeviceProps.C1_, count * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(C2_.data(), allVerticesDeviceProps.C2_, count * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(Cm_.data(), allVerticesDeviceProps.Cm_, count * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(I0_.data(), allVerticesDeviceProps.I0_, count * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(Iinject_.data(), allVerticesDeviceProps.Iinject_,
                           count * sizeof(BGFLOAT), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(Inoise_.data(), allVerticesDeviceProps.Inoise_, count * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(Isyn_.data(), allVerticesDeviceProps.Isyn_, count * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(Rm_.data(), allVerticesDeviceProps.Rm_, count * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(Tau_.data(), allVerticesDeviceProps.Tau_, count * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(Trefract_.data(), allVerticesDeviceProps.Trefract_,
                           count * sizeof(BGFLOAT), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(Vinit_.data(), allVerticesDeviceProps.Vinit_, count * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(Vm_.data(), allVerticesDeviceProps.Vm_, count * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(Vreset_.data(), allVerticesDeviceProps.Vreset_, count * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(Vrest_.data(), allVerticesDeviceProps.Vrest_, count * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(Vthresh_.data(), allVerticesDeviceProps.Vthresh_,
                           count * sizeof(BGFLOAT), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(numStepsInRefractoryPeriod_.data(),
                           allVerticesDeviceProps.numStepsInRefractoryPeriod_, count * sizeof(int),
                           cudaMemcpyDeviceToHost));
}
///  Clear the spike counts out of all neurons.
///
///  @param  allVerticesDevice   GPU address of the AllIFNeuronsDeviceProperties struct on device memory.
// TODO: Move this into EventBuffer somehow
void AllIFNeurons::clearNeuronSpikeCounts(void *allVerticesDevice)
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
