/**
 *      @file IAllNeurons.h
 *
 *      @brief An interface for neuron classes.
 */

#pragma once

using namespace std;

#include <iostream>

#include "Core/SynapseIndexMap.h"

class IAllSynapses;

class Layout;

class IAllNeurons {
public:
   virtual ~IAllNeurons() {}

   /**
    *  Setup the internal structure of the class.
    *  Allocate memories to store all neurons' state.
    *
    */
   virtual void setupNeurons() = 0;


    /*
     *  Load member variables from configuration file.
     *  Registered to OperationManager as Operation::loadParameters
     */
   virtual void loadParameters() = 0;

   /**
    *  Prints out all parameters of the neurons to logging file.
    *  Registered to OperationManager as Operation::printParameters
    */
   virtual void printParameters() const = 0;

   /**
    *  Creates all the Neurons and assigns initial data for them.
    *
    *  @param  layout      Layout information of the neunal network.
    */
   virtual void createAllNeurons(Layout *layout) = 0;

   /**
    *  Outputs state of the neuron chosen as a string.
    *
    *  @param  i   index of the neuron (in neurons) to output info from.
    *  @return the complete state of the neuron.
    */
   virtual string toString(const int i) const = 0;

#if defined(USE_GPU)
   public:
       /**
        *  Allocate GPU memories to store all neurons' states,
        *  and copy them from host to GPU memory.
        *
        *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
        */
       virtual void allocNeuronDeviceStruct(void** allNeuronsDevice) = 0;

       /**
        *  Delete GPU memories.
        *
        *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
        */
       virtual void deleteNeuronDeviceStruct(void* allNeuronsDevice) = 0;

       /**
        *  Copy all neurons' data from host to device.
        *
        *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
        */
       virtual void copyNeuronHostToDevice(void* allNeuronsDevice) = 0;

       /**
        *  Copy all neurons' data from device to host.
        *
        *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
        */
       virtual void copyNeuronDeviceToHost(void* allNeuronsDevice) = 0;

       /**
        *  Update the state of all neurons for a time step
        *  Notify outgoing synapses if neuron has fired.
        *
        *  @param  synapses               Reference to the allSynapses struct on host memory.
        *  @param  allNeuronsDevice       Reference to the allNeurons struct on device memory.
        *  @param  allSynapsesDevice      Reference to the allSynapses struct on device memory.
        *  @param  randNoise              Reference to the random noise array.
        *  @param  synapseIndexMapDevice  Reference to the SynapseIndexMap on device memory.
        */
       virtual void advanceNeurons(IAllSynapses &synapses, void* allNeuronsDevice, void* allSynapsesDevice, float* randNoise, SynapseIndexMap* synapseIndexMapDevice) = 0;

       /**
        *  Set some parameters used for advanceNeuronsDevice.
        *
        *  @param  synapses               Reference to the allSynapses struct on host memory.
        */
       virtual void setAdvanceNeuronsDeviceParams(IAllSynapses &synapses) = 0;
#else // !defined(USE_GPU)
public:
   /**
    *  Update internal state of the indexed Neuron (called by every simulation step).
    *  Notify outgoing synapses if neuron has fired.
    *
    *  @param  synapses         The Synapse list to search from.
    *  @param  synapseIndexMap  Reference to the SynapseIndexMap.
    */
   virtual void advanceNeurons(IAllSynapses &synapses, const SynapseIndexMap *synapseIndexMap) = 0;

#endif // defined(USE_GPU)
};
