/**
 * @file AllSynapsesDeviceFuncs.h
 * 
 * @ingroup Simulator/Edges
 *
 * @brief Device functions for synapse data
 */

#pragma once

#include "AllSpikingSynapses.h"
#include "AllSpikingNeurons.h"
#include "AllDSSynapses.h"
#include "AllSTDPSynapses.h"
#include "BGTypes.h"

using namespace std;

#if defined(__CUDACC__)

extern __device__ enumClassSynapses classSynapses_d = undefClassSynapses;

///  CUDA code for advancing spiking synapses.
///  Perform updating synapses for one time step.
///
///  @param[in] totalSynapseCount  Number of synapses.
///  @param  edgeIndexMapDevice    GPU address of the EdgeIndexMap on device memory.
///  @param[in] simulationStep        The current simulation step.
///  @param[in] deltaT                Inner simulation step duration.
///  @param[in] allEdgesDevice     Pointer to Synapse structures in device memory.
extern __global__ void advanceSpikingSynapsesDevice ( int totalSynapseCount, EdgeIndexMap* edgeIndexMapDevice, uint64_t simulationStep, const BGFLOAT deltaT, AllSpikingSynapsesDeviceProperties* allEdgesDevice );

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
extern __global__ void advanceSTDPSynapsesDevice ( int totalSynapseCount, EdgeIndexMap* edgeIndexMapDevice, uint64_t simulationStep, const BGFLOAT deltaT, AllSTDPSynapsesDeviceProperties* allEdgesDevice, AllSpikingNeuronsDeviceProperties* allVerticesDevice, int maxSpikes );

/// Adjust the strength of the synapse or remove it from the synapse map if it has gone below
/// zero.
///
/// @param[in] numVertices        Number of vertices.
/// @param[in] deltaT             The time step size.
/// @param[in] W_d                Array of synapse weight.
/// @param[in] maxEdges        Maximum number of synapses per neuron.
/// @param[in] allVerticesDevice   Pointer to the Neuron structures in device memory.
/// @param[in] allEdgesDevice  Pointer to the Synapse structures in device memory.
extern __global__ void updateSynapsesWeightsDevice( int numVertices, BGFLOAT deltaT, BGFLOAT* W_d, int maxEdges, AllSpikingNeuronsDeviceProperties* allVerticesDevice, AllSpikingSynapsesDeviceProperties* allEdgesDevice, vertexType* neuronTypeMapD );

/// Adds a synapse to the network.  Requires the locations of the source and
/// destination neurons.
///
/// @param allEdgesDevice      Pointer to the Synapse structures in device memory.
/// @param pSummationMap          Pointer to the summation point.
/// @param deltaT                 The simulation time step size.
/// @param weight                 Synapse weight.
extern __global__ void initSynapsesDevice( int n, AllDSSynapsesDeviceProperties* allEdgesDevice, BGFLOAT *pSummationMap, const BGFLOAT deltaT, BGFLOAT weight );

#endif 
