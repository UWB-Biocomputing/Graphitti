/**
 * @file AllSynapsesDeviceFuncs.h
 * 
 * @ingroup Simulation/Edges
 *
 * @brief 
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
///  @param  synapseIndexMapDevice    GPU address of the SynapseIndexMap on device memory.
///  @param[in] simulationStep        The current simulation step.
///  @param[in] deltaT                Inner simulation step duration.
///  @param[in] allSynapsesDevice     Pointer to Synapse structures in device memory.
extern __global__ void advanceSpikingSynapsesDevice ( int totalSynapseCount, SynapseIndexMap* synapseIndexMapDevice, uint64_t simulationStep, const BGFLOAT deltaT, AllSpikingSynapsesDeviceProperties* allSynapsesDevice );

///  CUDA code for advancing STDP synapses.
///  Perform updating synapses for one time step.
///
///  @param[in] totalSynapseCount  Number of synapses.
///  @param  synapseIndexMapDevice    GPU address of the SynapseIndexMap on device memory.
///  @param[in] simulationStep        The current simulation step.
///  @param[in] deltaT                Inner simulation step duration.
///  @param[in] allSynapsesDevice     Pointer to AllSTDPSynapsesDeviceProperties structures 
///                                   on device memory.
///  @param[in] allNeuronsDevice      GPU address of AllNeurons structures on device memory.
///  @param[in] maxSpikes             Maximum number of spikes per neuron per epoch.   
extern __global__ void advanceSTDPSynapsesDevice ( int totalSynapseCount, SynapseIndexMap* synapseIndexMapDevice, uint64_t simulationStep, const BGFLOAT deltaT, AllSTDPSynapsesDeviceProperties* allSynapsesDevice, AllSpikingNeuronsDeviceProperties* allNeuronsDevice, int maxSpikes );

/// Adjust the strength of the synapse or remove it from the synapse map if it has gone below
/// zero.
///
/// @param[in] numNeurons        Number of neurons.
/// @param[in] deltaT             The time step size.
/// @param[in] W_d                Array of synapse weight.
/// @param[in] maxSynapses        Maximum number of synapses per neuron.
/// @param[in] allNeuronsDevice   Pointer to the Neuron structures in device memory.
/// @param[in] allSynapsesDevice  Pointer to the Synapse structures in device memory.
extern __global__ void updateSynapsesWeightsDevice( int numNeurons, BGFLOAT deltaT, BGFLOAT* W_d, int maxSynapses, AllSpikingNeuronsDeviceProperties* allNeuronsDevice, AllSpikingSynapsesDeviceProperties* allSynapsesDevice, neuronType* neuronTypeMapD );

/// Adds a synapse to the network.  Requires the locations of the source and
/// destination neurons.
///
/// @param allSynapsesDevice      Pointer to the Synapse structures in device memory.
/// @param pSummationMap          Pointer to the summation point.
/// @param deltaT                 The simulation time step size.
/// @param weight                 Synapse weight.
extern __global__ void initSynapsesDevice( int n, AllDSSynapsesDeviceProperties* allSynapsesDevice, BGFLOAT *pSummationMap, const BGFLOAT deltaT, BGFLOAT weight );

#endif 
