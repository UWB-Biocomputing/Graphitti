/**
 *      @file IAllSynapses.h
 *
 *      @brief An interface for synapse classes.
 */

#pragma once

#include "Global.h"
#include "Core/Simulator.h"
#include "Simulation/Core/SynapseIndexMap.h"

class IAllNeurons;

typedef void (*fpCreateSynapse_t)(void*, const int, const int, int, int, BGFLOAT*, const BGFLOAT, synapseType);

// enumerate all non-abstract synapse classes.
enum enumClassSynapses {classAllSpikingSynapses, classAllDSSynapses, classAllSTDPSynapses, classAllDynamicSTDPSynapses, undefClassSynapses};

class IAllSynapses {
public:
   virtual ~IAllSynapses() {};

   /**
    *  Setup the internal structure of the class (allocate memories and initialize them).
    *
    */
   virtual void setupSynapses() = 0;

   /**
    *  Reset time varying state vars and recompute decay.
    *
    *  @param  iSyn     Index of the synapse to set.
    *  @param  deltaT   Inner simulation step duration
    */
   virtual void resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT) = 0;

   /**
    * Load member variables from configuration file.
    * Registered to OperationManager as Operation::op::loadParameters
    */
   virtual void loadParameters() = 0;

   /**
    *  Prints out all parameters to logging file.
    *  Registered to OperationManager as Operation::printParameters
    */
   virtual void printParameters() const = 0;

   /**
    *  Adds a Synapse to the model, connecting two Neurons.
    *
    *  @param  iSyn        Index of the synapse to be added.
    *  @param  type        The type of the Synapse to add.
    *  @param  srcNeuron  The Neuron that sends to this Synapse.
    *  @param  destNeuron The Neuron that receives from the Synapse.
    *  @param  sumPoint   Summation point address.
    *  @param  deltaT      Inner simulation step duration
    */
   virtual void
   addSynapse(BGSIZE &iSyn, synapseType type, const int srcNeuron, const int destNeuron, BGFLOAT *sumPoint,
              const BGFLOAT deltaT) = 0;

   /**
    *  Create a Synapse and connect it to the model.
    *
    *  @param  iSyn        Index of the synapse to set.
    *  @param  srcNeuron      Coordinates of the source Neuron.
    *  @param  destNeuron        Coordinates of the destination Neuron.
    *  @param  sumPoint   Summation point address.
    *  @param  deltaT      Inner simulation step duration.
    *  @param  type        Type of the Synapse to create.
    */
   virtual void createSynapse(const BGSIZE iSyn, int srcNeuron, int destNeuron, BGFLOAT *sumPoint, const BGFLOAT deltaT,
                              synapseType type) = 0;

   /**
    *  Create a synapse index map.
    *
    */
   virtual SynapseIndexMap *createSynapseIndexMap() = 0;

   /**
    *  Get the sign of the synapseType.
    *
    *  @param    type    synapseType I to I, I to E, E to I, or E to E
    *  @return   1 or -1, or 0 if error
    */
   virtual int synSign(const synapseType type) = 0;

   /**
    *  Prints SynapsesProps data to console.
    */
   virtual void printSynapsesProps() const = 0;

#if defined(USE_GPU)
   public:
       /**
        *  Allocate GPU memories to store all synapses' states,
        *  and copy them from host to GPU memory.
        *
        *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
        */
       virtual void allocSynapseDeviceStruct(void** allSynapsesDevice) = 0;

       /**
        *  Allocate GPU memories to store all synapses' states,
        *  and copy them from host to GPU memory.
        *
        *  @param  allSynapsesDevice     Reference to the allSynapses struct on device memory.
        *  @param  numNeurons           Number of neurons.
        *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
        */
       virtual void allocSynapseDeviceStruct( void** allSynapsesDevice, int numNeurons, int maxSynapsesPerNeuron ) = 0;

       /**
        *  Delete GPU memories.
        *
        *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
        */
       virtual void deleteSynapseDeviceStruct( void* allSynapsesDevice ) = 0;

       /**
        *  Copy all synapses' data from host to device.
        *
        *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
        */
       virtual void copySynapseHostToDevice(void* allSynapsesDevice) = 0;

       /**
        *  Copy all synapses' data from host to device.
        *
        *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
        *  @param  numNeurons           Number of neurons.
        *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
        */
       virtual void copySynapseHostToDevice( void* allSynapsesDevice, int numNeurons, int maxSynapsesPerNeuron ) = 0;

       /**
        *  Copy all synapses' data from device to host.
        *
        *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
        */
       virtual void copySynapseDeviceToHost( void* allSynapsesDevice) = 0;

       /**
        *  Get synapse_counts in AllSynapses struct on device memory.
        *
        *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
        */
       virtual void copyDeviceSynapseCountsToHost(void* allSynapsesDevice) = 0;

       /**
        *  Get summationCoord and in_use in AllSynapses struct on device memory.
        *
        *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
        */
       virtual void copyDeviceSynapseSumIdxToHost(void* allSynapsesDevice) = 0;

       /**
        *  Advance all the Synapses in the simulation.
        *  Update the state of all synapses for a time step.
        *
        *  @param  allSynapsesDevice      Reference to the allSynapses struct on device memory.
        *  @param  allNeuronsDevice       Reference to the allNeurons struct on device memory.
        *  @param  synapseIndexMapDevice  Reference to the SynapseIndexMap on device memory.
        */
       virtual void advanceSynapses(void* allSynapsesDevice, void* allNeuronsDevice, void* synapseIndexMapDevice) = 0;

       /**
        *  Set some parameters used for advanceSynapsesDevice.
        */
       virtual void setAdvanceSynapsesDeviceParams() = 0;

       /**
        *  Set synapse class ID defined by enumClassSynapses for the caller's Synapse class.
        *  The class ID will be set to classSynapses_d in device memory,
        *  and the classSynapses_d will be referred to call a device function for the
        *  particular synapse class.
        *  Because we cannot use virtual function (Polymorphism) in device functions,
        *  we use this scheme.
        *  Note: we used to use a function pointer; however, it caused the growth_cuda crash
        *  (see issue#137).
        */
       virtual void setSynapseClassID() = 0;

       /**
        *  Prints GPU SynapsesProps data.
        *
        *  @param  allSynapsesDeviceProps   Reference to the corresponding SynapsesDeviceProperties struct on device memory.
        */
       virtual void printGPUSynapsesProps( void* allSynapsesDeviceProps ) const = 0;

#else // !defined(USE_GPU)
public:
   /**
    *  Advance all the Synapses in the simulation.
    *  Update the state of all synapses for a time step.
    *
    *  @param  neurons   The Neuron list to search from.
    *  @param  synapseIndexMap   Pointer to SynapseIndexMap structure.
    */
   virtual void advanceSynapses(IAllNeurons *neurons, SynapseIndexMap *synapseIndexMap) = 0;

   /**
    *  Advance one specific Synapse.
    *
    *  @param  iSyn      Index of the Synapse to connect to.
    *  @param  neurons   The Neuron list to search from.
    */
   virtual void advanceSynapse(const BGSIZE iSyn, IAllNeurons *neurons) = 0;

   /**
    *  Remove a synapse from the network.
    *
    *  @param  neuronIndex   Index of a neuron to remove from.
    *  @param  iSyn           Index of a synapse to remove.
    */
   virtual void eraseSynapse(const int neuronIndex, const BGSIZE iSyn) = 0;

#endif // defined(USE_GPU)
};
