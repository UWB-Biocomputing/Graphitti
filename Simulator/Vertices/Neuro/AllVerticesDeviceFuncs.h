/**
 * @file AllVerticesDeviceFuncs.h
 * 
 * @ingroup Simulator/Vertices
 *
 * @brief Device functions for vertices
 */

#pragma once

#include "AllIZHNeurons.h"
#include "AllSTDPSynapses.h"
#include "AllSynapsesDeviceFuncs.h"

#if defined(__CUDACC__)

///  CUDA code for advancing LIF neurons
///
///  @param[in] totalVertices          Number of vertices.
///  @param[in] maxEdges           Maximum number of synapses per neuron.
///  @param[in] maxSpikes             Maximum number of spikes per neuron per epoch.
///  @param[in] deltaT                Inner simulation step duration.
///  @param[in] simulationStep        The current simulation step.
///  @param[in] randNoise             Pointer to device random noise array.
///  @param[in] allVerticesDevice      Pointer to Neuron structures in device memory.
///  @param[in] allEdgesDevice     Pointer to Synapse structures in device memory.
///  @param[in] edgeIndexMap       Inverse map, which is a table indexed by an input neuron and maps to the synapses that provide input to that neuron.
///  @param[in] fAllowBackPropagation True if back propagaion is allowed.
extern __global__ void advanceLIFNeuronsDevice( int totalVertices, int maxEdges, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, float* randNoise, AllIFNeuronsDeviceProperties* allVerticesDevice, AllSpikingSynapsesDeviceProperties* allEdgesDevice, EdgeIndexMap* edgeIndexMapDevice, bool fAllowBackPropagation );

///  CUDA code for advancing izhikevich neurons
///
///  @param[in] totalVertices          Number of vertices.
///  @param[in] maxEdges           Maximum number of synapses per neuron.
///  @param[in] maxSpikes             Maximum number of spikes per neuron per epoch.
///  @param[in] deltaT                Inner simulation step duration.
///  @param[in] simulationStep        The current simulation step.
///  @param[in] randNoise             Pointer to device random noise array.
///  @param[in] allVerticesDevice      Pointer to Neuron structures in device memory.
///  @param[in] allEdgesDevice     Pointer to Synapse structures in device memory.
///  @param[in] edgeIndexMap       Inverse map, which is a table indexed by an input neuron and maps to the synapses that provide input to that neuron.
///  @param[in] fAllowBackPropagation True if back propagaion is allowed.
extern __global__ void advanceIZHNeuronsDevice( int totalVertices, int maxEdges, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, float* randNoise, AllIZHNeuronsDeviceProperties* allVerticesDevice, AllSpikingSynapsesDeviceProperties* allEdgesDevice, EdgeIndexMap* edgeIndexMapDevice, bool fAllowBackPropagation );

#endif
