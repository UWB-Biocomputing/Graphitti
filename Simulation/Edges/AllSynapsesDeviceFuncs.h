#pragma once

#include "AllSpikingSynapses.h"
#include "AllSpikingNeurons.h"
#include "AllDSSynapses.h"
#include "AllSTDPSynapses.h"
#include "BGTypes.h"

using namespace std;


// SynapseIndexMapDevice struct used to replicate the CPU's SynapseIndexMap. GPU side doesn't support
// std::vectors so in order to use the SynapseIndexMap on the GPU we use arrays 
struct SynapseIndexMapDevice {
   /// Pointer to the outgoing synapse index map.
   BGSIZE* outgoingSynapseIndexMap_;

   /// The beginning index of the outgoing spiking synapse of each neuron.
   /// Indexed by a source neuron index.
   BGSIZE *outgoingSynapseBegin_;

   /// The number of outgoing synapses of each neuron.
   /// Indexed by a source neuron index.
   BGSIZE *outgoingSynapseCount_;

   /// Pointer to the incoming synapse index map.
   BGSIZE* incomingSynapseIndexMap_;

   /// The beginning index of the incoming spiking synapse of each neuron.
   /// Indexed by a destination neuron index.
   BGSIZE *incomingSynapseBegin_;

   /// The number of incoming synapses of each neuron.
   /// Indexed by a destination neuron index.
   BGSIZE *incomingSynapseCount_;

   SynapseIndexMapDevice() : numOfNeurons_(0), numOfSynapses_(0) {
      outgoingSynapseIndexMap_ = NULL;
      outgoingSynapseBegin_ = NULL;
      outgoingSynapseCount_ = NULL;

      incomingSynapseIndexMap_ = NULL;
      incomingSynapseBegin_ = NULL;
      incomingSynapseCount_ = NULL;
   };

   SynapseIndexMapDevice(int neuronCount, int synapseCount) : numOfNeurons_(neuronCount), numOfSynapses_(synapseCount) {
      outgoingSynapseIndexMap_ = new BGSIZE[synapseCount];
      outgoingSynapseBegin_ = new BGSIZE[neuronCount];
      outgoingSynapseCount_ = new BGSIZE[neuronCount];

      incomingSynapseIndexMap_ = new BGSIZE[synapseCount];
      incomingSynapseBegin_ = new BGSIZE[neuronCount];
      incomingSynapseCount_ = new BGSIZE[neuronCount];
   };

   ~SynapseIndexMapDevice() {
      if (numOfNeurons_ != 0) {
         delete[] outgoingSynapseBegin_;
         delete[] outgoingSynapseCount_;
         delete[] incomingSynapseBegin_;
         delete[] incomingSynapseCount_;
      }
      if (numOfSynapses_ != 0) {
         delete[] outgoingSynapseIndexMap_;
         delete[] incomingSynapseIndexMap_;
      }
   }

private:
    /// Number of total neurons.
    BGSIZE numOfNeurons_;

    /// Number of total active synapses.
    BGSIZE numOfSynapses_;
};



#if defined(__CUDACC__)

extern __device__ enumClassSynapses classSynapses_d;

/**
 *  CUDA code for advancing spiking synapses.
 *  Perform updating synapses for one time step.
 *
 *  @param[in] totalSynapseCount  Number of synapses.
 *  @param  synapseIndexMapDevice    Reference to the SynapseIndexMap on device memory.
 *  @param[in] simulationStep        The current simulation step.
 *  @param[in] deltaT                Inner simulation step duration.
 *  @param[in] allSynapsesDevice     Pointer to Synapse structures in device memory.
 */
extern __global__ void advanceSpikingSynapsesDevice ( int totalSynapseCount, SynapseIndexMapDevice* synapseIndexMapDevice, uint64_t simulationStep, const BGFLOAT deltaT, AllSpikingSynapsesDeviceProperties* allSynapsesDevice );

/*
 *  CUDA code for advancing STDP synapses.
 *  Perform updating synapses for one time step.
 *
 *  @param[in] totalSynapseCount  Number of synapses.
 *  @param  synapseIndexMapDevice    Reference to the SynapseIndexMap on device memory.
 *  @param[in] simulationStep        The current simulation step.
 *  @param[in] deltaT                Inner simulation step duration.
 *  @param[in] allSynapsesDevice     Pointer to AllSTDPSynapsesDeviceProperties structures 
 *                                   on device memory.
 */
extern __global__ void advanceSTDPSynapsesDevice ( int totalSynapseCount, SynapseIndexMapDevice* synapseIndexMapDevice, uint64_t simulationStep, const BGFLOAT deltaT, AllSTDPSynapsesDeviceProperties* allSynapsesDevice, AllSpikingNeuronsDeviceProperties* allNeuronsDevice, int maxSpikes, int width );

/**
 * Adjust the strength of the synapse or remove it from the synapse map if it has gone below
 * zero.
 *
 * @param[in] numNeurons        Number of neurons.
 * @param[in] deltaT             The time step size.
 * @param[in] W_d                Array of synapse weight.
 * @param[in] maxSynapses        Maximum number of synapses per neuron.
 * @param[in] allNeuronsDevice   Pointer to the Neuron structures in device memory.
 * @param[in] allSynapsesDevice  Pointer to the Synapse structures in device memory.
 */
extern __global__ void updateSynapsesWeightsDevice( int numNeurons, BGFLOAT deltaT, BGFLOAT* W_d, int maxSynapses, AllSpikingNeuronsDeviceProperties* allNeuronsDevice, AllSpikingSynapsesDeviceProperties* allSynapsesDevice, neuronType* neuronTypeMapD );

/** 
 * Adds a synapse to the network.  Requires the locations of the source and
 * destination neurons.
 *
 * @param allSynapsesDevice      Pointer to the Synapse structures in device memory.
 * @param pSummationMap          Pointer to the summation point.
 * @param width                  Width of neuron map (assumes square).
 * @param deltaT                 The simulation time step size.
 * @param weight                 Synapse weight.
 */
extern __global__ void initSynapsesDevice( int n, AllDSSynapsesDeviceProperties* allSynapsesDevice, BGFLOAT *pSummationMap, int width, const BGFLOAT deltaT, BGFLOAT weight );

#endif 
