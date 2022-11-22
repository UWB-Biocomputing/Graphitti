/**
 * @file AllSTDPSynapses_d.cpp
 * 
 * @ingroup Simulator/Edges
 *
 * @brief A container of all STDP synapse data
 */

#include "AllSTDPSynapses.h"
#include "AllSpikingSynapses.h"
#include "AllSynapsesDeviceFuncs.h"
#include "Book.h"
#include "GPUModel.h"

///  CUDA code for advancing STDP synapses.
///  Perform updating synapses for one time step.
///
///  @param[in] totalSynapseCount  Number of synapses.
///  @param  edgeIndexMapDevice    GPU address of the EdgeIndexMap on device memory.
///  @param[in] simulationStep        The current simulation step.
///  @param[in] deltaT                Inner simulation step duration.
///  @param[in] allEdgesDevice     Pointer to AllSTDPSynapsesDeviceProperties structures
///                                   on device memory.
///  @param[in] allVerticesDevice      GPU address of AllNeurons structures on device memory.
///  @param[in] maxSpikes             Maximum number of spikes per neuron per epoch.

__global__ void advanceSTDPSynapsesDevice(int totalSynapseCount, EdgeIndexMap *edgeIndexMapDevice,
                                          uint64_t simulationStep, const BGFLOAT deltaT,
                                          AllSTDPSynapsesDeviceProperties *allEdgesDevice,
                                          AllSpikingNeuronsDeviceProperties *allVerticesDevice,
                                          int maxSpikes);


///  Allocate GPU memories to store all synapses' states,
///  and copy them from host to GPU memory.
///
///  @param  allEdgesDevice  GPU address of the AllSTDPSynapsesDeviceProperties struct
///                             on device memory.
void AllSTDPSynapses::allocEdgeDeviceStruct(void **allEdgesDevice)
{
   allocEdgeDeviceStruct(allEdgesDevice, Simulator::getInstance().getTotalVertices(),
                         Simulator::getInstance().getMaxEdgesPerVertex());
}

///  Allocate GPU memories to store all synapses' states,
///  and copy them from host to GPU memory.
///
///  @param  allEdgesDevice     GPU address of the AllSTDPSynapsesDeviceProperties struct
///                                on device memory.
///  @param  numVertices            Number of vertices.
///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
void AllSTDPSynapses::allocEdgeDeviceStruct(void **allEdgesDevice, int numVertices,
                                            int maxEdgesPerVertex)
{
   AllSTDPSynapsesDeviceProperties allEdgesDeviceProps;

   allocDeviceStruct(allEdgesDeviceProps, numVertices, maxEdgesPerVertex);

   HANDLE_ERROR(cudaMalloc(allEdgesDevice, sizeof(AllSTDPSynapsesDeviceProperties)));
   HANDLE_ERROR(cudaMemcpy(*allEdgesDevice, &allEdgesDeviceProps,
                           sizeof(AllSTDPSynapsesDeviceProperties), cudaMemcpyHostToDevice));
}

///  Allocate GPU memories to store all synapses' states,
///  and copy them from host to GPU memory.
///  (Helper function of allocEdgeDeviceStruct)
///
///  @param  allEdgesDevice     GPU address of the AllSTDPSynapsesDeviceProperties struct
///                                on device memory.
///  @param  numVertices            Number of vertices.
///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
void AllSTDPSynapses::allocDeviceStruct(AllSTDPSynapsesDeviceProperties &allEdgesDevice,
                                        int numVertices, int maxEdgesPerVertex)
{
   AllSpikingSynapses::allocDeviceStruct(allEdgesDevice, numVertices, maxEdgesPerVertex);

   BGSIZE maxTotalSynapses = maxEdgesPerVertex * numVertices;

   HANDLE_ERROR(
      cudaMalloc((void **)&allEdgesDevice.totalDelayPost_, maxTotalSynapses * sizeof(int)));
   HANDLE_ERROR(
      cudaMalloc((void **)&allEdgesDevice.delayQueuePost_, maxTotalSynapses * sizeof(BGSIZE)));
   HANDLE_ERROR(
      cudaMalloc((void **)&allEdgesDevice.delayIndexPost_, maxTotalSynapses * sizeof(int)));
   HANDLE_ERROR(
      cudaMalloc((void **)&allEdgesDevice.delayQueuePost_, maxTotalSynapses * sizeof(int)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.tauspost_, maxTotalSynapses * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.tauspre_, maxTotalSynapses * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.taupos_, maxTotalSynapses * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.tauneg_, maxTotalSynapses * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.STDPgap_, maxTotalSynapses * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.Wex_, maxTotalSynapses * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.Aneg_, maxTotalSynapses * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.Apos_, maxTotalSynapses * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.mupos_, maxTotalSynapses * sizeof(BGFLOAT)));
   HANDLE_ERROR(cudaMalloc((void **)&allEdgesDevice.muneg_, maxTotalSynapses * sizeof(BGFLOAT)));
}

///  Delete GPU memories.
///
///  @param  allEdgesDevice  GPU address of the AllSTDPSynapsesDeviceProperties struct
///                             on device memory.
void AllSTDPSynapses::deleteEdgeDeviceStruct(void *allEdgesDevice)
{
   AllSTDPSynapsesDeviceProperties allEdgesDeviceProps;

   HANDLE_ERROR(cudaMemcpy(&allEdgesDeviceProps, allEdgesDevice,
                           sizeof(AllSTDPSynapsesDeviceProperties), cudaMemcpyDeviceToHost));

   deleteDeviceStruct(allEdgesDeviceProps);

   HANDLE_ERROR(cudaFree(allEdgesDevice));
}

///  Delete GPU memories.
///  (Helper function of deleteEdgeDeviceStruct)
///
///  @param  allEdgesDevice  GPU address of the AllSTDPSynapsesDeviceProperties struct
///                             on device memory.
void AllSTDPSynapses::deleteDeviceStruct(AllSTDPSynapsesDeviceProperties &allEdgesDevice)
{
   HANDLE_ERROR(cudaFree(allEdgesDevice.totalDelayPost_));
   HANDLE_ERROR(cudaFree(allEdgesDevice.delayQueuePost_));
   HANDLE_ERROR(cudaFree(allEdgesDevice.delayIndexPost_));
   HANDLE_ERROR(cudaFree(allEdgesDevice.tauspost_));
   HANDLE_ERROR(cudaFree(allEdgesDevice.tauspre_));
   HANDLE_ERROR(cudaFree(allEdgesDevice.taupos_));
   HANDLE_ERROR(cudaFree(allEdgesDevice.tauneg_));
   HANDLE_ERROR(cudaFree(allEdgesDevice.STDPgap_));
   HANDLE_ERROR(cudaFree(allEdgesDevice.Wex_));
   HANDLE_ERROR(cudaFree(allEdgesDevice.Aneg_));
   HANDLE_ERROR(cudaFree(allEdgesDevice.Apos_));
   HANDLE_ERROR(cudaFree(allEdgesDevice.mupos_));
   HANDLE_ERROR(cudaFree(allEdgesDevice.muneg_));

   AllSpikingSynapses::deleteDeviceStruct(allEdgesDevice);
}

///  Copy all synapses' data from host to device.
///
///  @param  allEdgesDevice     GPU address of the AllSTDPSynapsesDeviceProperties struct
///                                on device memory.
///  @param  numVertices            Number of vertices.
///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
void AllSTDPSynapses::copyEdgeHostToDevice(void *allEdgesDevice)
{   // copy everything necessary
   copyEdgeHostToDevice(allEdgesDevice, Simulator::getInstance().getTotalVertices(),
                        Simulator::getInstance().getMaxEdgesPerVertex());
}

///  Copy all synapses' data from host to device.
///
///  @param  allEdgesDevice     GPU address of the AllSTDPSynapsesDeviceProperties struct
///                                on device memory.
///  @param  numVertices            Number of vertices.
///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
void AllSTDPSynapses::copyEdgeHostToDevice(void *allEdgesDevice, int numVertices,
                                           int maxEdgesPerVertex)
{   // copy everything necessary
   AllSTDPSynapsesDeviceProperties allEdgesDeviceProps;

   HANDLE_ERROR(cudaMemcpy(&allEdgesDeviceProps, allEdgesDevice,
                           sizeof(AllSTDPSynapsesDeviceProperties), cudaMemcpyDeviceToHost));

   copyHostToDevice(allEdgesDevice, allEdgesDeviceProps, numVertices, maxEdgesPerVertex);
}

///  Copy all synapses' data from host to device.
///  (Helper function of copyEdgeHostToDevice)
///
///  @param  allEdgesDevice     GPU address of the AllSTDPSynapsesDeviceProperties struct
///                                on device memory.
///  @param  numVertices            Number of vertices.
///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
void AllSTDPSynapses::copyHostToDevice(void *allEdgesDevice,
                                       AllSTDPSynapsesDeviceProperties &allEdgesDeviceProps,
                                       int numVertices, int maxEdgesPerVertex)
{   // copy everything necessary
   AllSpikingSynapses::copyHostToDevice(allEdgesDevice, allEdgesDeviceProps, numVertices,
                                        maxEdgesPerVertex);

   BGSIZE maxTotalSynapses = maxEdgesPerVertex * numVertices;

   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.totalDelayPost_, totalDelayPost_,
                           maxTotalSynapses * sizeof(int), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.delayQueuePost_, delayQueuePost_,
                           maxTotalSynapses * sizeof(uint32_t), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.delayIndexPost_, delayIndexPost_,
                           maxTotalSynapses * sizeof(int), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.delayQueuePost_, delayQueuePost_,
                           maxTotalSynapses * sizeof(int), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.tauspost_, tauspost_,
                           maxTotalSynapses * sizeof(BGFLOAT), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.tauspre_, tauspre_,
                           maxTotalSynapses * sizeof(BGFLOAT), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.taupos_, taupos_, maxTotalSynapses * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.tauneg_, tauneg_, maxTotalSynapses * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.STDPgap_, STDPgap_,
                           maxTotalSynapses * sizeof(BGFLOAT), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.Wex_, Wex_, maxTotalSynapses * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.Aneg_, Aneg_, maxTotalSynapses * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.Apos_, Apos_, maxTotalSynapses * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.mupos_, mupos_, maxTotalSynapses * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(allEdgesDeviceProps.muneg_, muneg_, maxTotalSynapses * sizeof(BGFLOAT),
                           cudaMemcpyHostToDevice));
}

///  Copy all synapses' data from device to host.
///
///  @param  allEdgesDevice  GPU address of the AllSTDPSynapsesDeviceProperties struct
///                             on device memory.
void AllSTDPSynapses::copyEdgeDeviceToHost(void *allEdgesDevice)
{
   // copy everything necessary
   AllSTDPSynapsesDeviceProperties allEdgesDeviceProps;

   HANDLE_ERROR(cudaMemcpy(&allEdgesDeviceProps, allEdgesDevice,
                           sizeof(AllSTDPSynapsesDeviceProperties), cudaMemcpyDeviceToHost));

   copyDeviceToHost(allEdgesDeviceProps);
}

///  Copy all synapses' data from device to host.
///  (Helper function of copyEdgeDeviceToHost)
///
///  @param  allEdgesDevice     GPU address of the AllSTDPSynapsesDeviceProperties struct
///                                on device memory.
///  @param  numVertices            Number of vertices.
///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
void AllSTDPSynapses::copyDeviceToHost(AllSTDPSynapsesDeviceProperties &allEdgesDevice)
{
   AllSpikingSynapses::copyDeviceToHost(allEdgesDevice);

   int numVertices = Simulator::getInstance().getTotalVertices();
   BGSIZE maxTotalSynapses = Simulator::getInstance().getMaxEdgesPerVertex() * numVertices;

   HANDLE_ERROR(cudaMemcpy(delayQueuePost_, allEdgesDevice.delayQueuePost_,
                           maxTotalSynapses * sizeof(uint32_t), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(delayIndexPost_, allEdgesDevice.delayIndexPost_,
                           maxTotalSynapses * sizeof(int), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(delayQueuePost_, allEdgesDevice.delayQueuePost_,
                           maxTotalSynapses * sizeof(int), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(tauspost_, allEdgesDevice.tauspost_, maxTotalSynapses * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(tauspre_, allEdgesDevice.tauspre_, maxTotalSynapses * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(taupos_, allEdgesDevice.taupos_, maxTotalSynapses * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(tauneg_, allEdgesDevice.tauneg_, maxTotalSynapses * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(STDPgap_, allEdgesDevice.STDPgap_, maxTotalSynapses * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(Wex_, allEdgesDevice.Wex_, maxTotalSynapses * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(Aneg_, allEdgesDevice.Aneg_, maxTotalSynapses * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(Apos_, allEdgesDevice.Apos_, maxTotalSynapses * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(mupos_, allEdgesDevice.mupos_, maxTotalSynapses * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(muneg_, allEdgesDevice.muneg_, maxTotalSynapses * sizeof(BGFLOAT),
                           cudaMemcpyDeviceToHost));
}

///  Advance all the Synapses in the simulation.
///  Update the state of all synapses for a time step.
///
///  @param  allEdgesDevice      GPU address of the AllEdgesDeviceProperties struct
///                                 on device memory.
///  @param  allVerticesDevice       GPU address of the allNeurons struct on device memory.
///  @param  edgeIndexMapDevice  GPU address of the EdgeIndexMap on device memory.
void AllSTDPSynapses::advanceEdges(void *allEdgesDevice, void *allVerticesDevice,
                                   void *edgeIndexMapDevice)
{
   int maxSpikes = (int)((Simulator::getInstance().getEpochDuration()
                          * Simulator::getInstance().getMaxFiringRate()));

   // CUDA parameters
   const int threadsPerBlock = 256;
   int blocksPerGrid = (totalEdgeCount_ + threadsPerBlock - 1) / threadsPerBlock;
   // Advance synapses ------------->
   advanceSTDPSynapsesDevice<<<blocksPerGrid, threadsPerBlock>>>(
      totalEdgeCount_, (EdgeIndexMap *)edgeIndexMapDevice, g_simulationStep,
      Simulator::getInstance().getDeltaT(), (AllSTDPSynapsesDeviceProperties *)allEdgesDevice,
      (AllSpikingNeuronsDeviceProperties *)allVerticesDevice, maxSpikes);
}

///  Set synapse class ID defined by enumClassSynapses for the caller's Synapse class.
///  The class ID will be set to classSynapses_d in device memory,
///  and the classSynapses_d will be referred to call a device function for the
///  particular synapse class.
///  Because we cannot use virtual function (Polymorphism) in device functions,
///  we use this scheme.
///  Note: we used to use a function pointer; however, it caused the growth_cuda crash
///  (see issue#137).
void AllSTDPSynapses::setEdgeClassID()
{
   enumClassSynapses classSynapses_h = classAllSTDPSynapses;

   HANDLE_ERROR(cudaMemcpyToSymbol(classSynapses_d, &classSynapses_h, sizeof(enumClassSynapses)));
}

///  Prints GPU SynapsesProps data.
///
///  @param  allEdgesDeviceProps   GPU address of the corresponding SynapsesDeviceProperties struct on device memory.
void AllSTDPSynapses::printGPUEdgesProps(void *allEdgesDeviceProps) const
{
   AllSTDPSynapsesDeviceProperties allSynapsesProps;

   //allocate print out data members
   BGSIZE size = maxEdgesPerVertex_ * countVertices_;
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

      int *totalDelayPostPrint = new int[size];
      BGFLOAT *tauspostPrint = new BGFLOAT[size];
      BGFLOAT *tausprePrint = new BGFLOAT[size];
      BGFLOAT *tauposPrint = new BGFLOAT[size];
      BGFLOAT *taunegPrint = new BGFLOAT[size];
      BGFLOAT *STDPgapPrint = new BGFLOAT[size];
      BGFLOAT *WexPrint = new BGFLOAT[size];
      BGFLOAT *AnegPrint = new BGFLOAT[size];
      BGFLOAT *AposPrint = new BGFLOAT[size];
      BGFLOAT *muposPrint = new BGFLOAT[size];
      BGFLOAT *munegPrint = new BGFLOAT[size];

      // copy everything
      HANDLE_ERROR(cudaMemcpy(&allSynapsesProps, allEdgesDeviceProps,
                              sizeof(AllSTDPSynapsesDeviceProperties), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(synapseCountsPrint, allSynapsesProps.edgeCounts_,
                              countVertices_ * sizeof(BGSIZE), cudaMemcpyDeviceToHost));
      maxEdgesPerVertexPrint = allSynapsesProps.maxEdgesPerVertex_;
      totalSynapseCountPrint = allSynapsesProps.totalEdgeCount_;
      countNeuronsPrint = allSynapsesProps.countVertices_;

      // Set countVertices_ to 0 to avoid illegal memory deallocation
      // at AllSynapsesProps deconstructor.
      allSynapsesProps.countVertices_ = 0;

      HANDLE_ERROR(cudaMemcpy(sourceNeuronIndexPrint, allSynapsesProps.sourceVertexIndex_,
                              size * sizeof(int), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(destNeuronIndexPrint, allSynapsesProps.destVertexIndex_,
                              size * sizeof(int), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(
         cudaMemcpy(WPrint, allSynapsesProps.W_, size * sizeof(BGFLOAT), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(typePrint, allSynapsesProps.type_, size * sizeof(edgeType),
                              cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(psrPrint, allSynapsesProps.psr_, size * sizeof(BGFLOAT),
                              cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(inUsePrint, allSynapsesProps.inUse_, size * sizeof(bool),
                              cudaMemcpyDeviceToHost));

      HANDLE_ERROR(cudaMemcpy(decayPrint, allSynapsesProps.decay_, size * sizeof(BGFLOAT),
                              cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(tauPrint, allSynapsesProps.tau_, size * sizeof(BGFLOAT),
                              cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(totalDelayPrint, allSynapsesProps.totalDelay_, size * sizeof(int),
                              cudaMemcpyDeviceToHost));

      HANDLE_ERROR(cudaMemcpy(totalDelayPostPrint, allSynapsesProps.totalDelayPost_,
                              size * sizeof(int), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(tauspostPrint, allSynapsesProps.tauspost_, size * sizeof(BGFLOAT),
                              cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(tausprePrint, allSynapsesProps.tauspre_, size * sizeof(BGFLOAT),
                              cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(tauposPrint, allSynapsesProps.taupos_, size * sizeof(BGFLOAT),
                              cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(taunegPrint, allSynapsesProps.tauneg_, size * sizeof(BGFLOAT),
                              cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(STDPgapPrint, allSynapsesProps.STDPgap_, size * sizeof(BGFLOAT),
                              cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(WexPrint, allSynapsesProps.Wex_, size * sizeof(BGFLOAT),
                              cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(AnegPrint, allSynapsesProps.Aneg_, size * sizeof(BGFLOAT),
                              cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(AposPrint, allSynapsesProps.Apos_, size * sizeof(BGFLOAT),
                              cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(muposPrint, allSynapsesProps.mupos_, size * sizeof(BGFLOAT),
                              cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(munegPrint, allSynapsesProps.muneg_, size * sizeof(BGFLOAT),
                              cudaMemcpyDeviceToHost));

      for (int i = 0; i < maxEdgesPerVertex_ * countVertices_; i++) {
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
            cout << " GPU tauspost_: " << tauspostPrint[i];
            cout << " GPU tauspre_: " << tausprePrint[i];
            cout << " GPU taupos_: " << tauposPrint[i];
            cout << " GPU tauneg_: " << taunegPrint[i];
            cout << " GPU STDPgap_: " << STDPgapPrint[i];
            cout << " GPU Wex_: " << WexPrint[i];
            cout << " GPU Aneg_: " << AnegPrint[i];
            cout << " GPU Apos_: " << AposPrint[i];
            cout << " GPU mupos_: " << muposPrint[i];
            cout << " GPU muneg_: " << munegPrint[i] << endl;
         }
      }

      for (int i = 0; i < countVertices_; i++) {
         cout << "GPU edge_counts:"
              << "neuron[" << i << "]" << synapseCountsPrint[i] << endl;
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

      delete[] totalDelayPostPrint;
      delete[] tauspostPrint;
      delete[] tausprePrint;
      delete[] tauposPrint;
      delete[] taunegPrint;
      delete[] STDPgapPrint;
      delete[] WexPrint;
      delete[] AnegPrint;
      delete[] AposPrint;
      delete[] muposPrint;
      delete[] munegPrint;
      totalDelayPostPrint = nullptr;
      tauspostPrint = nullptr;
      tausprePrint = nullptr;
      tauposPrint = nullptr;
      taunegPrint = nullptr;
      STDPgapPrint = nullptr;
      WexPrint = nullptr;
      AnegPrint = nullptr;
      AposPrint = nullptr;
      muposPrint = nullptr;
      munegPrint = nullptr;
   }
}


///  Adjust synapse weight according to the Spike-timing-dependent synaptic modification
///  induced by natural spike trains
///
///  @param  allEdgesDevice    GPU address of the AllSTDPSynapsesDeviceProperties structures
///                               on device memory.
///  @param  iEdg                 Index of the synapse to set.
///  @param  delta                Pre/post synaptic spike interval.
///  @param  epost                Params for the rule given in Froemke and Dan (2002).
///  @param  epre                 Params for the rule given in Froemke and Dan (2002).
CUDA_CALLABLE void stdpLearningDevice(AllSTDPSynapsesDeviceProperties *allEdgesDevice,
                                      const BGSIZE iEdg, double delta, double epost, double epre)
{
   BGFLOAT STDPgap = allEdgesDevice->STDPgap_[iEdg];
   BGFLOAT muneg = allEdgesDevice->muneg_[iEdg];
   BGFLOAT mupos = allEdgesDevice->mupos_[iEdg];
   BGFLOAT tauneg = allEdgesDevice->tauneg_[iEdg];
   BGFLOAT taupos = allEdgesDevice->taupos_[iEdg];
   BGFLOAT Aneg = allEdgesDevice->Aneg_[iEdg];
   BGFLOAT Apos = allEdgesDevice->Apos_[iEdg];
   BGFLOAT Wex = allEdgesDevice->Wex_[iEdg];
   BGFLOAT &W = allEdgesDevice->W_[iEdg];
   edgeType type = allEdgesDevice->type_[iEdg];
   BGFLOAT dw;

   if (delta < -STDPgap) {
      // Depression
      dw = pow(fabs(W) / Wex, muneg) * Aneg * exp(delta / tauneg);   // normalize
   } else if (delta > STDPgap) {
      // Potentiation
      dw = pow(fabs(Wex - fabs(W)) / Wex, mupos) * Apos * exp(-delta / taupos);   // normalize
   } else {
      return;
   }

   // dw is the percentage change in synaptic strength; add 1.0 to become the scaling ratio
   dw = 1.0 + dw * epre * epost;

   // if scaling ratio is less than zero, set it to zero so this synapse, its strength is always zero
   if (dw < 0) {
      dw = 0;
   }

   // current weight multiplies dw (scaling ratio) to generate new weight
   W *= dw;

   // if new weight is bigger than Wex_ (maximum allowed weight), then set it to Wex_
   if (fabs(W) > Wex) {
      W = edgSign(type) * Wex;
   }

   // DEBUG_SYNAPSE(
   //     printf("AllSTDPSynapses::stdpLearning:\n");
   //     printf("          iEdg: %d\n", iEdg);
   //     printf("          delta: %f\n", delta);
   //     printf("          epre: %f\n", epre);
   //     printf("          epost: %f\n", epost);
   //     printf("          dw: %f\n", dw);
   //     printf("          W: %f\n\n", W);
   // );
}

///  Checks if there is an input spike in the queue.
///
///  @param[in] allEdgesDevice     GPU adress of AllSTDPSynapsesDeviceProperties structures
///                                   on device memory.
///  @param[in] iEdg                  Index of the Synapse to check.
///
///  @return true if there is an input spike event.
CUDA_CALLABLE bool isSTDPSynapseSpikeQueuePostDevice(AllSTDPSynapsesDeviceProperties *allEdgesDevice, BGSIZE iEdg)
{
   uint32_t &delayQueue = allEdgesDevice->delayQueuePost_[iEdg];
   int &delayIndex = allEdgesDevice->delayIndexPost_[iEdg];
   int delayQueueLength = allEdgesDevice->delayQueuePostLength_[iEdg];

   uint32_t delayMask = (0x1 << delayIndex);
   bool isFired = delayQueue & (delayMask);
   delayQueue &= ~(delayMask);
   if (++delayIndex >= delayQueueLength) {
      delayIndex = 0;
   }
   return isFired;
}

///  Gets the spike history of the neuron.
///
///  @param  allVerticesDevice       GPU address of the allNeurons struct on device memory.
///  @param  index                  Index of the neuron to get spike history.
///  @param  offIndex               Offset of the history beffer to get.
///                                 -1 will return the last spike.
///  @param  maxSpikes              Maximum number of spikes per neuron per epoch.
///
///  @return Spike history.
CUDA_CALLABLE uint64_t getSTDPSynapseSpikeHistoryDevice(
   AllSpikingNeuronsDeviceProperties *allVerticesDevice, int index, int offIndex, int maxSpikes)
{
      int idxSp = allVerticesDevice->queueEnd_[index] + offIndex;
      if (idxSp < 0) 
         idxSp = idxSp + maxSpikes;
      return allVerticesDevice->spikeHistory_[index][idxSp];
}

///  CUDA code for advancing STDP synapses.
///  Perform updating synapses for one time step.
///
///  @param[in] totalSynapseCount        Number of synapses.
///  @param[in] edgeIndexMapDevice    GPU address of the EdgeIndexMap on device memory.
///  @param[in] simulationStep           The current simulation step.
///  @param[in] deltaT                   Inner simulation step duration.
///  @param[in] allEdgesDevice        GPU address of AllSTDPSynapsesDeviceProperties structures
///                                      on device memory.
///  @param[in] allVerticesDevice         GPU address of AllVertices structures on device memory.
///  @param[in] maxSpikes                Maximum number of spikes per neuron per epoch.
__global__ void advanceSTDPSynapsesDevice(int totalSynapseCount, EdgeIndexMap *edgeIndexMapDevice,
                                          uint64_t simulationStep, const BGFLOAT deltaT,
                                          AllSTDPSynapsesDeviceProperties *allEdgesDevice,
                                          AllSpikingNeuronsDeviceProperties *allVerticesDevice,
                                          int maxSpikes)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx >= totalSynapseCount)
      return;

   BGSIZE iEdg = edgeIndexMapDevice->incomingEdgeIndexMap_[idx];

   // If the synapse is inhibitory or its weight is zero, update synapse state using AllSpikingSynapses::advanceEdge method
   BGFLOAT &W = allEdgesDevice->W_[iEdg];
   if (W <= 0.0) {
      BGFLOAT &psr = allEdgesDevice->psr_[iEdg];
      BGFLOAT decay = allEdgesDevice->decay_[iEdg];

      // Checks if there is an input spike in the queue.
      bool isFired = isSpikingSynapsesSpikeQueueDevice(allEdgesDevice, iEdg);

      // is an input in the queue?
      if (isFired) {
         switch (classSynapses_d) {
            case classAllSTDPSynapses:
               changeSpikingSynapsesPSRDevice(
                  static_cast<AllSpikingSynapsesDeviceProperties *>(allEdgesDevice), iEdg,
                  simulationStep, deltaT);
               break;
            case classAllDynamicSTDPSynapses:
               // Note: we cast void * over the allEdgesDevice, then recast it,
               // because AllDSSynapsesDeviceProperties inherited properties from
               // the AllDSSynapsesDeviceProperties and the AllSTDPSynapsesDeviceProperties.
               changeDSSynapsePSRDevice(
                  static_cast<AllDSSynapsesDeviceProperties *>((void *)allEdgesDevice), iEdg,
                  simulationStep, deltaT);
               break;
            default:
               assert(false);
         }
      }
      // decay the post spike response
      psr *= decay;
      return;
   }

   BGFLOAT &decay = allEdgesDevice->decay_[iEdg];
   BGFLOAT &psr = allEdgesDevice->psr_[iEdg];

   // is an input in the queue?
   bool fPre = isSpikingSynapsesSpikeQueueDevice(allEdgesDevice, iEdg);
   bool fPost = isSTDPSynapseSpikeQueuePostDevice(allEdgesDevice, iEdg);
   if (fPre || fPost) {
      BGFLOAT &tauspre = allEdgesDevice->tauspre_[iEdg];
      BGFLOAT &tauspost = allEdgesDevice->tauspost_[iEdg];
      BGFLOAT &taupos = allEdgesDevice->taupos_[iEdg];
      BGFLOAT &tauneg = allEdgesDevice->tauneg_[iEdg];
      int &totalDelay = allEdgesDevice->totalDelay_[iEdg];
      bool &useFroemkeDanSTDP = allEdgesDevice->useFroemkeDanSTDP_[iEdg];

      // pre and post neurons index
      int idxPre = allEdgesDevice->sourceVertexIndex_[iEdg];
      int idxPost = allEdgesDevice->destVertexIndex_[iEdg];
      int64_t spikeHistory, spikeHistory2;
      BGFLOAT delta;
      BGFLOAT epre, epost;

      if (fPre) {   // preSpikeHit
         // spikeCount points to the next available position of spike_history,
         // so the getSpikeHistory w/offset = -2 will return the spike time
         // just one before the last spike.
         spikeHistory = getSTDPSynapseSpikeHistoryDevice(allVerticesDevice, idxPre, -2, maxSpikes);
         if (spikeHistory > 0 && useFroemkeDanSTDP) {
            // delta will include the transmission delay
            delta = static_cast<BGFLOAT>(simulationStep - spikeHistory) * deltaT;
            epre = 1.0 - exp(-delta / tauspre);
         } else {
            epre = 1.0;
         }

         // call the learning function stdpLearning() for each pair of
         // pre-post spikes
         int offIndex = -1;   // last spike
         while (true) {
            spikeHistory
               = getSTDPSynapseSpikeHistoryDevice(allVerticesDevice, idxPost, offIndex, maxSpikes);
            if (spikeHistory == ULONG_MAX)
               break;
            // delta is the spike interval between pre-post spikes
            delta = -static_cast<BGFLOAT>(simulationStep - spikeHistory) * deltaT;

            if (delta <= -3.0 * tauneg)
               break;
            if (useFroemkeDanSTDP) {
               spikeHistory2 = getSTDPSynapseSpikeHistoryDevice(allVerticesDevice, idxPost,
                                                                offIndex - 1, maxSpikes);
               if (spikeHistory2 == ULONG_MAX)
                  break;
               epost = 1.0
                       - exp(-(static_cast<BGFLOAT>(spikeHistory - spikeHistory2) * deltaT)
                             / tauspost);
            } else {
               epost = 1.0;
            }
            stdpLearningDevice(allEdgesDevice, iEdg, delta, epost, epre);
            --offIndex;
         }

         switch (classSynapses_d) {
            case classAllSTDPSynapses:
               changeSpikingSynapsesPSRDevice(
                  static_cast<AllSpikingSynapsesDeviceProperties *>(allEdgesDevice), iEdg,
                  simulationStep, deltaT);
               break;
            case classAllDynamicSTDPSynapses:
               // Note: we cast void * over the allEdgesDevice, then recast it,
               // because AllDSSynapsesDeviceProperties inherited properties from
               // the AllDSSynapsesDeviceProperties and the AllSTDPSynapsesDeviceProperties.
               changeDSSynapsePSRDevice(
                  static_cast<AllDSSynapsesDeviceProperties *>((void *)allEdgesDevice), iEdg,
                  simulationStep, deltaT);
               break;
            default:
               assert(false);
         }
      }

      if (fPost) {   // postSpikeHit
         // spikeCount points to the next available position of spike_history,
         // so the getSpikeHistory w/offset = -2 will return the spike time
         // just one before the last spike.
         spikeHistory = getSTDPSynapseSpikeHistoryDevice(allVerticesDevice, idxPost, -2, maxSpikes);
         if (spikeHistory > 0 && useFroemkeDanSTDP) {
            // delta will include the transmission delay
            delta = static_cast<BGFLOAT>(simulationStep - spikeHistory) * deltaT;
            epost = 1.0 - exp(-delta / tauspost);
         } else {
            epost = 1.0;
         }

         // call the learning function stdpLearning() for each pair of
         // post-pre spikes
         int offIndex = -1;   // last spike
         while (true) {
            spikeHistory
               = getSTDPSynapseSpikeHistoryDevice(allVerticesDevice, idxPre, offIndex, maxSpikes);
            if (spikeHistory == ULONG_MAX)
               break;

            if (spikeHistory + totalDelay > simulationStep) {
               --offIndex;
               continue;
            }
            // delta is the spike interval between post-pre spikes
            delta = static_cast<BGFLOAT>(simulationStep - spikeHistory - totalDelay) * deltaT;


            if (delta >= 3.0 * taupos)
               break;
            if (useFroemkeDanSTDP) {
               spikeHistory2 = getSTDPSynapseSpikeHistoryDevice(allVerticesDevice, idxPre,
                                                                offIndex - 1, maxSpikes);
               if (spikeHistory2 == ULONG_MAX)
                  break;
               epre
                  = 1.0
                    - exp(-(static_cast<BGFLOAT>(spikeHistory - spikeHistory2) * deltaT) / tauspre);
            } else {
               epre = 1.0;
            }
            stdpLearningDevice(allEdgesDevice, iEdg, delta, epost, epre);
            --offIndex;
         }
      }
   }

   // decay the post spike response
   psr *= decay;
}
