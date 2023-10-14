/**
 * @file AllSynapsesDeviceFuncs.h
 * 
 * @ingroup Simulator/Edges
 *
 * @brief Device functions for synapse data
 */

#pragma once

#include "AllDSSynapses.h"
#include "AllSTDPSynapses.h"
#include "AllSpikingNeurons.h"
#include "AllSpikingSynapses.h"
#include "BGTypes.h"

using namespace std;

#if defined(__CUDACC__)

extern CUDA_CALLABLE enumClassSynapses classSynapses_d;
extern CUDA_CALLABLE int edgSign(edgeType t);
extern CUDA_CALLABLE void changeDSSynapsePSRDevice(AllDSSynapsesDeviceProperties *allEdgesDevice,
                                                   BGSIZE iEdg, const uint64_t simulationStep,
                                                   BGFLOAT deltaT);


/// Adjust the strength of the synapse or remove it from the synapse map if it has gone below
/// zero.
///
/// @param[in] numVertices        Number of vertices.
/// @param[in] deltaT             The time step size.
/// @param[in] W_d                Array of synapse weight.
/// @param[in] maxEdges        Maximum number of synapses per neuron.
/// @param[in] allVerticesDevice   Pointer to the Neuron structures in device memory.
/// @param[in] allEdgesDevice  Pointer to the Synapse structures in device memory.
extern __global__ void
   updateSynapsesWeightsDevice(int numVertices, BGFLOAT deltaT, BGFLOAT *W_d, int maxEdges,
                               AllSpikingNeuronsDeviceProperties *allVerticesDevice,
                               AllSpikingSynapsesDeviceProperties *allEdgesDevice,
                               vertexType *neuronTypeMapD);

/// Adds a synapse to the network.  Requires the locations of the source and
/// destination neurons.
///
/// @param allEdgesDevice      Pointer to the Synapse structures in device memory.
/// @param pSummationMap          Pointer to the summation point.
/// @param deltaT                 The simulation time step size.
/// @param weight                 Synapse weight.
extern __global__ void initSynapsesDevice(int n, AllDSSynapsesDeviceProperties *allEdgesDevice,
                                          BGFLOAT *pSummationMap, BGFLOAT deltaT, BGFLOAT weight);

#endif
