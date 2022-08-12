/**
 * @file ConnGrowth_d.cpp
 *
 * @ingroup Simulator/Connections
 *
 *
 * @brief Update the weights of the Synapses in the simulation.
 */

#include "AllSpikingSynapses.h"
#include "AllSynapsesDeviceFuncs.h"
#include "Book.h"
#include "ConnGrowth.h"
#include "Simulator.h"
#include "AllDynamicSTDPSynapses.h"
#include "AllSTDPSynapses.h"



__global__ void updateSynapsesWeightsDevice(
    int numVertices, BGFLOAT deltaT, BGFLOAT *W_d, int maxEdges,
    AllSpikingNeuronsDeviceProperties *allVerticesDevice,
    AllSpikingSynapsesDeviceProperties *allEdgesDevice,
    vertexType *neuronTypeMap_d);


CUDA_CALLABLE edgeType edgType(vertexType *neuronTypeMap_d, const int srcVertex,
                               const int destVertex);

CUDA_CALLABLE void
eraseSpikingSynapse(AllSpikingSynapsesDeviceProperties *allEdgesDevice,
                    const int neuronIndex, const int synapseOffset,
                    int maxEdges);

CUDA_CALLABLE void
addSpikingSynapse(AllSpikingSynapsesDeviceProperties *allEdgesDevice,
                  edgeType type, const int srcVertex, const int destVertex,
                  int sourceIndex, int destIndex, BGFLOAT *sumPoint,
                  const BGFLOAT deltaT, BGFLOAT *W_d, int numVertices);

CUDA_CALLABLE void
createDynamicSTDPSynapse(AllDynamicSTDPSynapsesDeviceProperties *allEdgesDevice,
                         const int neuronIndex, const int synapseOffset,
                         int sourceIndex, int destIndex, BGFLOAT *sumPoint,
                         const BGFLOAT deltaT, edgeType type);

CUDA_CALLABLE void
createSTDPSynapse(AllSTDPSynapsesDeviceProperties *allEdgesDevice,
                  const int neuronIndex, const int synapseOffset,
                  int sourceIndex, int destIndex, BGFLOAT *sumPoint,
                  const BGFLOAT deltaT, edgeType type);

CUDA_CALLABLE void
createDSSynapse(AllDSSynapsesDeviceProperties *allEdgesDevice,
                const int neuronIndex, const int synapseOffset, int sourceIndex,
                int destIndex, BGFLOAT *sumPoint, const BGFLOAT deltaT,
                edgeType type);

CUDA_CALLABLE void
createSpikingSynapse(AllSpikingSynapsesDeviceProperties *allEdgesDevice,
                     const int neuronIndex, const int synapseOffset,
                     int sourceIndex, int destIndex, BGFLOAT *sumPoint,
                     const BGFLOAT deltaT, edgeType type);

/*
 *  Update the weights of the Synapses in the simulation. To be clear,
 *  iterates through all source and destination vertices and updates their
 *  edge strengths from the weight matrix.
 *  Note: Platform Dependent.
 *
 *  @param  numVertices         number of vertices to update.
 *  @param  vertices            The AllVertices object.
 *  @param  synapses           The AllEdges object.
 *  @param  allVerticesDevice   GPU address to the AllVertices struct in device
 * memory.
 *  @param  allEdgesDevice  GPU address to the allEdges struct in device memory.
 *  @param  layout             The Layout object.
 */
void ConnGrowth::updateSynapsesWeights(
    const int numVertices, AllVertices &vertices, AllEdges &synapses,
    AllSpikingNeuronsDeviceProperties *allVerticesDevice,
    AllSpikingSynapsesDeviceProperties *allEdgesDevice, Layout *layout) {
  Simulator &simulator = Simulator::getInstance();
  // For now, we just set the weights to equal the areas. We will later
  // scale it and set its sign (when we index and get its sign).
  (*W_) = (*area_);

  BGFLOAT deltaT = simulator.getDeltaT();

  // CUDA parameters
  const int threadsPerBlock = 256;
  int blocksPerGrid;

  // allocate device memories
  BGSIZE W_d_size = simulator.getTotalVertices() *
                    simulator.getTotalVertices() * sizeof(BGFLOAT);
  BGFLOAT *W_h = new BGFLOAT[W_d_size];
  BGFLOAT *W_d;
  HANDLE_ERROR(cudaMalloc((void **)&W_d, W_d_size));

  vertexType *neuronTypeMapD;
  HANDLE_ERROR(cudaMalloc((void **)&neuronTypeMapD,
                          simulator.getTotalVertices() * sizeof(vertexType)));

  // copy weight data to the device memory
  for (int i = 0; i < simulator.getTotalVertices(); i++)
    for (int j = 0; j < simulator.getTotalVertices(); j++)
      W_h[i * simulator.getTotalVertices() + j] = (*W_)(i, j);

  HANDLE_ERROR(cudaMemcpy(W_d, W_h, W_d_size, cudaMemcpyHostToDevice));

  HANDLE_ERROR(cudaMemcpy(neuronTypeMapD, layout->vertexTypeMap_,
                          simulator.getTotalVertices() * sizeof(vertexType),
                          cudaMemcpyHostToDevice));

  blocksPerGrid =
      (simulator.getTotalVertices() + threadsPerBlock - 1) / threadsPerBlock;
  updateSynapsesWeightsDevice<<<blocksPerGrid, threadsPerBlock>>>(
      simulator.getTotalVertices(), deltaT, W_d,
      simulator.getMaxEdgesPerVertex(), allVerticesDevice, allEdgesDevice,
      neuronTypeMapD);

  // free memories
  HANDLE_ERROR(cudaFree(W_d));
  delete[] W_h;

  HANDLE_ERROR(cudaFree(neuronTypeMapD));

  // copy device synapse count to host memory
  synapses.copyDeviceEdgeCountsToHost(allEdgesDevice);
  // copy device synapse summation coordinate to host memory
  synapses.copyDeviceEdgeSumIdxToHost(allEdgesDevice);
}



///@}

/******************************************
 * @name Global Functions for updateSynapses
 ******************************************/
///@{

/// Adjust the strength of the synapse or remove it from the synapse map if it
/// has gone below zero.
///
/// @param[in] numVertices        Number of vertices.
/// @param[in] deltaT             The time step size.
/// @param[in] W_d                Array of synapse weight.
/// @param[in] maxEdges        Maximum number of synapses per neuron.
/// @param[in] allVerticesDevice   Pointer to the Neuron structures in device
/// memory.
/// @param[in] allEdgesDevice  Pointer to the Synapse structures in device
/// memory.
__global__ void updateSynapsesWeightsDevice(
    int numVertices, BGFLOAT deltaT, BGFLOAT *W_d, int maxEdges,
    AllSpikingNeuronsDeviceProperties *allVerticesDevice,
    AllSpikingSynapsesDeviceProperties *allEdgesDevice,
    vertexType *neuronTypeMap_d) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numVertices)
    return;

  int adjusted = 0;
  // int could_have_been_removed = 0; // TODO: use this value
  int removed = 0;
  int added = 0;

  // Scale and add sign to the areas
  // visit each neuron 'a'
  int destVertex = idx;

  // and each destination neuron 'b'
  for (int srcVertex = 0; srcVertex < numVertices; srcVertex++) {
    // visit each synapse at (xa,ya)
    bool connected = false;
    edgeType type = edgType(neuronTypeMap_d, srcVertex, destVertex);

    // for each existing synapse
    BGSIZE existing_synapses = allEdgesDevice->edgeCounts_[destVertex];
    int existingSynapsesChecked = 0;
    for (BGSIZE synapseIndex = 0;
         (existingSynapsesChecked < existing_synapses) && !connected;
         synapseIndex++) {
      BGSIZE iEdg = maxEdges * destVertex + synapseIndex;
      if (allEdgesDevice->inUse_[iEdg] == true) {
        // if there is a synapse between a and b
        if (allEdgesDevice->sourceVertexIndex_[iEdg] == srcVertex) {
          connected = true;
          adjusted++;

          // adjust the strength of the synapse or remove
          // it from the synapse map if it has gone below
          // zero.
          if (W_d[srcVertex * numVertices + destVertex] <= 0) {
            removed++;
            eraseSpikingSynapse(allEdgesDevice, destVertex, synapseIndex,
                                maxEdges);
          } else {
            // adjust
            // g_synapseStrengthAdjustmentConstant is 1.0e-8;
            allEdgesDevice->W_[iEdg] =
                W_d[srcVertex * numVertices + destVertex] * edgSign(type) *
                AllNeuroEdges::SYNAPSE_STRENGTH_ADJUSTMENT;
          }
        }
        existingSynapsesChecked++;
      }
    }

    // if not connected and weight(a,b) > 0, add a new synapse from a to b
    if (!connected && (W_d[srcVertex * numVertices + destVertex] > 0)) {
      // locate summation point
      BGFLOAT *sumPoint = &(allVerticesDevice->summationMap_[destVertex]);
      added++;

      addSpikingSynapse(allEdgesDevice, type, srcVertex, destVertex, srcVertex,
                        destVertex, sumPoint, deltaT, W_d, numVertices);
    }
  }
}


/// Returns the type of synapse at the given coordinates
///
/// @param[in] allVerticesDevice          Pointer to the Neuron structures in
/// device memory.
/// @param srcVertex             Index of the source neuron.
/// @param destVertex            Index of the destination neuron.
CUDA_CALLABLE edgeType edgType(vertexType *neuronTypeMap_d, const int srcVertex,
                               const int destVertex) {
  if (neuronTypeMap_d[srcVertex] == INH && neuronTypeMap_d[destVertex] == INH)
    return II;
  else if (neuronTypeMap_d[srcVertex] == INH &&
           neuronTypeMap_d[destVertex] == EXC)
    return IE;
  else if (neuronTypeMap_d[srcVertex] == EXC &&
           neuronTypeMap_d[destVertex] == INH)
    return EI;
  else if (neuronTypeMap_d[srcVertex] == EXC &&
           neuronTypeMap_d[destVertex] == EXC)
    return EE;

  return ETYPE_UNDEF;
}




/// Remove a synapse from the network.
///
/// @param[in] allEdgesDevice      Pointer to the
/// AllSpikingSynapsesDeviceProperties structures
///                                   on device memory.
/// @param neuronIndex               Index of a neuron.
/// @param synapseOffset             Offset into neuronIndex's synapses.
/// @param[in] maxEdges            Maximum number of synapses per neuron.
CUDA_CALLABLE void
eraseSpikingSynapse(AllSpikingSynapsesDeviceProperties *allEdgesDevice,
                    const int neuronIndex, const int synapseOffset,
                    int maxEdges) {
  BGSIZE iSync = maxEdges * neuronIndex + synapseOffset;
  allEdgesDevice->edgeCounts_[neuronIndex]--;
  allEdgesDevice->inUse_[iSync] = false;
  allEdgesDevice->W_[iSync] = 0;
}



/// Adds a synapse to the network.  Requires the locations of the source and
/// destination neurons.
///
/// @param allEdgesDevice      Pointer to the AllSpikingSynapsesDeviceProperties
/// structures
///                               on device memory.
/// @param type                   Type of the Synapse to create.
/// @param srcVertex            Coordinates of the source Neuron.
/// @param destVertex           Coordinates of the destination Neuron.
///
/// @param sumPoint              Pointer to the summation point.
/// @param deltaT                 The time step size.
/// @param W_d                    Array of synapse weight.
/// @param numVertices            The number of vertices.
CUDA_CALLABLE void
addSpikingSynapse(AllSpikingSynapsesDeviceProperties *allEdgesDevice,
                  edgeType type, const int srcVertex, const int destVertex,
                  int sourceIndex, int destIndex, BGFLOAT *sumPoint,
                  const BGFLOAT deltaT, BGFLOAT *W_d, int numVertices) {
  if (allEdgesDevice->edgeCounts_[destVertex] >=
      allEdgesDevice->maxEdgesPerVertex_) {
    return; // TODO: ERROR!
  }

  // add it to the list
  BGSIZE synapseIndex;
  BGSIZE maxEdges = allEdgesDevice->maxEdgesPerVertex_;
  BGSIZE synapseBegin = maxEdges * destVertex;
  for (synapseIndex = 0; synapseIndex < maxEdges; synapseIndex++) {
    if (!allEdgesDevice->inUse_[synapseBegin + synapseIndex]) {
      break;
    }
  }

  allEdgesDevice->edgeCounts_[destVertex]++;

  // create a synapse
  switch (classSynapses_d) {
  case classAllSpikingSynapses:
    createSpikingSynapse(allEdgesDevice, destVertex, synapseIndex, sourceIndex,
                         destIndex, sumPoint, deltaT, type);
    break;
  case classAllDSSynapses:
    createDSSynapse(
        static_cast<AllDSSynapsesDeviceProperties *>(allEdgesDevice),
        destVertex, synapseIndex, sourceIndex, destIndex, sumPoint, deltaT,
        type);
    break;
  case classAllSTDPSynapses:
    createSTDPSynapse(
        static_cast<AllSTDPSynapsesDeviceProperties *>(allEdgesDevice),
        destVertex, synapseIndex, sourceIndex, destIndex, sumPoint, deltaT,
        type);
    break;
  case classAllDynamicSTDPSynapses:
    createDynamicSTDPSynapse(
        static_cast<AllDynamicSTDPSynapsesDeviceProperties *>(allEdgesDevice),
        destVertex, synapseIndex, sourceIndex, destIndex, sumPoint, deltaT,
        type);
    break;
  default:
    assert(false);
  }
  allEdgesDevice->W_[synapseBegin + synapseIndex] =
      W_d[srcVertex * numVertices + destVertex] * edgSign(type) *
      AllNeuroEdges::SYNAPSE_STRENGTH_ADJUSTMENT;
}



///@}

/******************************************
 * @name Device Functions for createEdge
 ******************************************/
///@{

///  Create a Synapse and connect it to the model.
///
///  @param allEdgesDevice    GPU address of the
///  AllDynamicSTDPSynapsesDeviceProperties structures
///                              on device memory.
///  @param neuronIndex          Index of the source neuron.
///  @param synapseOffset        Offset (into neuronIndex's) of the Synapse to
///  create.
///  @param srcVertex            Coordinates of the source Neuron.
///  @param destVertex           Coordinates of the destination Neuron.
///  @param sumPoint             Pointer to the summation point.
///  @param deltaT               The time step size.
///  @param type                 Type of the Synapse to create.
CUDA_CALLABLE void
createDynamicSTDPSynapse(AllDynamicSTDPSynapsesDeviceProperties *allEdgesDevice,
                         const int neuronIndex, const int synapseOffset,
                         int sourceIndex, int destIndex, BGFLOAT *sumPoint,
                         const BGFLOAT deltaT, edgeType type) {
  BGFLOAT delay;
  BGSIZE maxEdges = allEdgesDevice->maxEdgesPerVertex_;
  BGSIZE iEdg = maxEdges * neuronIndex + synapseOffset;

  allEdgesDevice->inUse_[iEdg] = true;
  allEdgesDevice->destVertexIndex_[iEdg] = destIndex;
  allEdgesDevice->sourceVertexIndex_[iEdg] = sourceIndex;
  allEdgesDevice->W_[iEdg] = edgSign(type) * 10.0e-9;

  allEdgesDevice->delayQueue_[iEdg] = 0;
  allEdgesDevice->delayIndex_[iEdg] = 0;
  allEdgesDevice->delayQueueLength_[iEdg] = LENGTH_OF_DELAYQUEUE;

  allEdgesDevice->psr_[iEdg] = 0.0;
  allEdgesDevice->r_[iEdg] = 1.0;
  allEdgesDevice->u_[iEdg] = 0.4; // DEFAULT_U
  allEdgesDevice->lastSpike_[iEdg] = ULONG_MAX;
  allEdgesDevice->type_[iEdg] = type;

  allEdgesDevice->U_[iEdg] = DEFAULT_U;
  allEdgesDevice->tau_[iEdg] = DEFAULT_tau;

  BGFLOAT U;
  BGFLOAT D;
  BGFLOAT F;
  BGFLOAT tau;
  switch (type) {
  case II:
    U = 0.32;
    D = 0.144;
    F = 0.06;
    tau = 6e-3;
    delay = 0.8e-3;
    break;
  case IE:
    U = 0.25;
    D = 0.7;
    F = 0.02;
    tau = 6e-3;
    delay = 0.8e-3;
    break;
  case EI:
    U = 0.05;
    D = 0.125;
    F = 1.2;
    tau = 3e-3;
    delay = 0.8e-3;
    break;
  case EE:
    U = 0.5;
    D = 1.1;
    F = 0.05;
    tau = 3e-3;
    delay = 1.5e-3;
    break;
  default:
    break;
  }

  allEdgesDevice->U_[iEdg] = U;
  allEdgesDevice->D_[iEdg] = D;
  allEdgesDevice->F_[iEdg] = F;

  allEdgesDevice->tau_[iEdg] = tau;
  allEdgesDevice->decay_[iEdg] = exp(-deltaT / tau);
  allEdgesDevice->totalDelay_[iEdg] = static_cast<int>(delay / deltaT) + 1;

  uint32_t size = allEdgesDevice->totalDelay_[iEdg] / (sizeof(uint8_t) * 8) + 1;
  assert(size <= BYTES_OF_DELAYQUEUE);

  // May 1st 2020
  // Use constants from Froemke and Dan (2002).
  // Spike-timing-dependent synaptic modification induced by natural spike
  // trains. Nature 416 (3/2002)
  allEdgesDevice->Apos_[iEdg] = 1.01;
  allEdgesDevice->Aneg_[iEdg] = -0.52;
  allEdgesDevice->STDPgap_[iEdg] = 2e-3;

  allEdgesDevice->totalDelayPost_[iEdg] = 0;

  allEdgesDevice->tauspost_[iEdg] = 75e-3;
  allEdgesDevice->tauspre_[iEdg] = 34e-3;

  allEdgesDevice->taupos_[iEdg] = 14.8e-3;
  allEdgesDevice->tauneg_[iEdg] = 33.8e-3;
  allEdgesDevice->Wex_[iEdg] = 5.0265e-7;

  allEdgesDevice->mupos_[iEdg] = 0;
  allEdgesDevice->muneg_[iEdg] = 0;

  allEdgesDevice->useFroemkeDanSTDP_[iEdg] = false;
}



///  Create a Synapse and connect it to the model.
///
///  @param allEdgesDevice    GPU address of the AllSTDPSynapsesDeviceProperties
///  structures
///                              on device memory.
///  @param neuronIndex          Index of the source neuron.
///  @param synapseOffset        Offset (into neuronIndex's) of the Synapse to
///  create.
///  @param srcVertex            Coordinates of the source Neuron.
///  @param destVertex           Coordinates of the destination Neuron.
///  @param sumPoint             Pointer to the summation point.
///  @param deltaT               The time step size.
///  @param type                 Type of the Synapse to create.
CUDA_CALLABLE void
createSTDPSynapse(AllSTDPSynapsesDeviceProperties *allEdgesDevice,
                  const int neuronIndex, const int synapseOffset,
                  int sourceIndex, int destIndex, BGFLOAT *sumPoint,
                  const BGFLOAT deltaT, edgeType type) {
  BGFLOAT delay;
  BGSIZE maxEdges = allEdgesDevice->maxEdgesPerVertex_;
  BGSIZE iEdg = maxEdges * neuronIndex + synapseOffset;

  allEdgesDevice->inUse_[iEdg] = true;
  allEdgesDevice->destVertexIndex_[iEdg] = destIndex;
  allEdgesDevice->sourceVertexIndex_[iEdg] = sourceIndex;
  allEdgesDevice->W_[iEdg] = edgSign(type) * 10.0e-9;

  allEdgesDevice->delayQueue_[iEdg] = 0;
  allEdgesDevice->delayIndex_[iEdg] = 0;
  allEdgesDevice->delayQueueLength_[iEdg] = LENGTH_OF_DELAYQUEUE;

  allEdgesDevice->psr_[iEdg] = 0.0;
  allEdgesDevice->type_[iEdg] = type;

  allEdgesDevice->tau_[iEdg] = DEFAULT_tau;

  BGFLOAT tau;
  switch (type) {
  case II:
    tau = 6e-3;
    delay = 0.8e-3;
    break;
  case IE:
    tau = 6e-3;
    delay = 0.8e-3;
    break;
  case EI:
    tau = 3e-3;
    delay = 0.8e-3;
    break;
  case EE:
    tau = 3e-3;
    delay = 1.5e-3;
    break;
  default:
    break;
  }

  allEdgesDevice->tau_[iEdg] = tau;
  allEdgesDevice->decay_[iEdg] = exp(-deltaT / tau);
  allEdgesDevice->totalDelay_[iEdg] = static_cast<int>(delay / deltaT) + 1;

  uint32_t size = allEdgesDevice->totalDelay_[iEdg] / (sizeof(uint8_t) * 8) + 1;
  assert(size <= BYTES_OF_DELAYQUEUE);

  // May 1st 2020
  // Use constants from Froemke and Dan (2002).
  // Spike-timing-dependent synaptic modification induced by natural spike
  // trains. Nature 416 (3/2002)
  allEdgesDevice->Apos_[iEdg] = 1.01;
  allEdgesDevice->Aneg_[iEdg] = -0.52;
  allEdgesDevice->STDPgap_[iEdg] = 2e-3;

  allEdgesDevice->totalDelayPost_[iEdg] = 0;

  allEdgesDevice->tauspost_[iEdg] = 75e-3;
  allEdgesDevice->tauspre_[iEdg] = 34e-3;

  allEdgesDevice->taupos_[iEdg] = 14.8e-3;
  allEdgesDevice->tauneg_[iEdg] = 33.8e-3;
  allEdgesDevice->Wex_[iEdg] = 5.0265e-7;

  allEdgesDevice->mupos_[iEdg] = 0;
  allEdgesDevice->muneg_[iEdg] = 0;

  allEdgesDevice->useFroemkeDanSTDP_[iEdg] = false;
}

///  Create a DS Synapse and connect it to the model.
///
///  @param allEdgesDevice    GPU address of the AllDSSynapsesDeviceProperties
///  structures
///                              on device memory.
///  @param neuronIndex          Index of the source neuron.
///  @param synapseOffset        Offset (into neuronIndex's) of the Synapse to
///  create.
///  @param srcVertex            Coordinates of the source Neuron.
///  @param destVertex           Coordinates of the destination Neuron.
///  @param sumPoint             Pointer to the summation point.
///  @param deltaT               The time step size.
///  @param type                 Type of the Synapse to create.
CUDA_CALLABLE void
createDSSynapse(AllDSSynapsesDeviceProperties *allEdgesDevice,
                const int neuronIndex, const int synapseOffset, int sourceIndex,
                int destIndex, BGFLOAT *sumPoint, const BGFLOAT deltaT,
                edgeType type) {
  BGFLOAT delay;
  BGSIZE maxEdges = allEdgesDevice->maxEdgesPerVertex_;
  BGSIZE iEdg = maxEdges * neuronIndex + synapseOffset;

  allEdgesDevice->inUse_[iEdg] = true;
  allEdgesDevice->destVertexIndex_[iEdg] = destIndex;
  allEdgesDevice->sourceVertexIndex_[iEdg] = sourceIndex;
  allEdgesDevice->W_[iEdg] = edgSign(type) * 10.0e-9;

  allEdgesDevice->delayQueue_[iEdg] = 0;
  allEdgesDevice->delayIndex_[iEdg] = 0;
  allEdgesDevice->delayQueueLength_[iEdg] = LENGTH_OF_DELAYQUEUE;

  allEdgesDevice->psr_[iEdg] = 0.0;
  allEdgesDevice->r_[iEdg] = 1.0;
  allEdgesDevice->u_[iEdg] = 0.4; // DEFAULT_U
  allEdgesDevice->lastSpike_[iEdg] = ULONG_MAX;
  allEdgesDevice->type_[iEdg] = type;

  allEdgesDevice->U_[iEdg] = DEFAULT_U;
  allEdgesDevice->tau_[iEdg] = DEFAULT_tau;

  BGFLOAT U;
  BGFLOAT D;
  BGFLOAT F;
  BGFLOAT tau;
  switch (type) {
  case II:
    U = 0.32;
    D = 0.144;
    F = 0.06;
    tau = 6e-3;
    delay = 0.8e-3;
    break;
  case IE:
    U = 0.25;
    D = 0.7;
    F = 0.02;
    tau = 6e-3;
    delay = 0.8e-3;
    break;
  case EI:
    U = 0.05;
    D = 0.125;
    F = 1.2;
    tau = 3e-3;
    delay = 0.8e-3;
    break;
  case EE:
    U = 0.5;
    D = 1.1;
    F = 0.05;
    tau = 3e-3;
    delay = 1.5e-3;
    break;
  default:
    break;
  }

  allEdgesDevice->U_[iEdg] = U;
  allEdgesDevice->D_[iEdg] = D;
  allEdgesDevice->F_[iEdg] = F;

  allEdgesDevice->tau_[iEdg] = tau;
  allEdgesDevice->decay_[iEdg] = exp(-deltaT / tau);
  allEdgesDevice->totalDelay_[iEdg] = static_cast<int>(delay / deltaT) + 1;

  uint32_t size = allEdgesDevice->totalDelay_[iEdg] / (sizeof(uint8_t) * 8) + 1;
  assert(size <= BYTES_OF_DELAYQUEUE);
}



///  Create a Spiking Synapse and connect it to the model.
///
///  @param allEdgesDevice    GPU address of the
///  AllSpikingSynapsesDeviceProperties structures
///                              on device memory.
///  @param neuronIndex          Index of the source neuron.
///  @param synapseOffset        Offset (into neuronIndex's) of the Synapse to
///  create.
///  @param srcVertex            Coordinates of the source Neuron.
///  @param destVertex           Coordinates of the destination Neuron.
///  @param sumPoint             Pointer to the summation point.
///  @param deltaT               The time step size.
///  @param type                 Type of the Synapse to create.
CUDA_CALLABLE void
createSpikingSynapse(AllSpikingSynapsesDeviceProperties *allEdgesDevice,
                     const int neuronIndex, const int synapseOffset,
                     int sourceIndex, int destIndex, BGFLOAT *sumPoint,
                     const BGFLOAT deltaT, edgeType type) {
  BGFLOAT delay;
  BGSIZE maxEdges = allEdgesDevice->maxEdgesPerVertex_;
  BGSIZE iEdg = maxEdges * neuronIndex + synapseOffset;

  allEdgesDevice->inUse_[iEdg] = true;
  allEdgesDevice->destVertexIndex_[iEdg] = destIndex;
  allEdgesDevice->sourceVertexIndex_[iEdg] = sourceIndex;
  allEdgesDevice->W_[iEdg] = edgSign(type) * 10.0e-9;

  allEdgesDevice->delayQueue_[iEdg] = 0;
  allEdgesDevice->delayIndex_[iEdg] = 0;
  allEdgesDevice->delayQueueLength_[iEdg] = LENGTH_OF_DELAYQUEUE;

  allEdgesDevice->psr_[iEdg] = 0.0;
  allEdgesDevice->type_[iEdg] = type;

  allEdgesDevice->tau_[iEdg] = DEFAULT_tau;

  BGFLOAT tau;
  switch (type) {
  case II:
    tau = 6e-3;
    delay = 0.8e-3;
    break;
  case IE:
    tau = 6e-3;
    delay = 0.8e-3;
    break;
  case EI:
    tau = 3e-3;
    delay = 0.8e-3;
    break;
  case EE:
    tau = 3e-3;
    delay = 1.5e-3;
    break;
  default:
    break;
  }

  allEdgesDevice->tau_[iEdg] = tau;
  allEdgesDevice->decay_[iEdg] = exp(-deltaT / tau);
  allEdgesDevice->totalDelay_[iEdg] = static_cast<int>(delay / deltaT) + 1;

  uint32_t size = allEdgesDevice->totalDelay_[iEdg] / (sizeof(uint8_t) * 8) + 1;
  assert(size <= BYTES_OF_DELAYQUEUE);
}


// /******************************************
//  * @name Device Functions for utility
//  ******************************************/
// ///@{

// /// Return 1 if originating neuron is excitatory, -1 otherwise.
// ///
// /// @param[in] t  edgeType I to I, I to E, E to I, or E to E
// /// @return 1 or -1
// CUDA_CALLABLE int edgSign(edgeType t) {
//   switch (t) {
//   case II:
//   case IE:
//     return -1;
//   case EI:
//   case EE:
//     return 1;
//   }

//   return 0;
// }
