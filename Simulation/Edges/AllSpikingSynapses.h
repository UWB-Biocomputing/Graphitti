/**
 *      @file AllSpikingSynapses.h
 *
 *      @brief A container of all spiking synapse data
 */

/** 
 * @authors Aaron Oziel, Sean Blackbourn
 * 
 * @class AllSpikingSynapses AllSpikingSynapses.h "AllSpikingSynapses.h"
 *
 * \latexonly  \subsubsection*{Implementation} \endlatexonly
 * \htmlonly   <h3>Implementation</h3> \endhtmlonly
 *
 *  The container holds synapse parameters of all synapses. 
 *  Each kind of synapse parameter is stored in a 2D array. Each item in the first 
 *  dimention of the array corresponds with each neuron, and each item in the second
 *  dimension of the array corresponds with a synapse parameter of each synapse of the neuron. 
 *  Bacause each neuron owns different number of synapses, the number of synapses 
 *  for each neuron is stored in a 1D array, synapse_counts.
 *
 *  For CUDA implementation, we used another structure, AllSynapsesDevice, where synapse
 *  parameters are stored in 1D arrays instead of 2D arrays, so that device functions
 *  can access these data less latency. When copying a synapse parameter, P[i][j],
 *  from host to device, it is stored in P[i * max_synapses_per_neuron + j] in 
 *  AllSynapsesDevice structure.
 *
 *  The latest implementation uses the identical data struture between host and CUDA;
 *  that is, synapse parameters are stored in a 1D array, so we don't need conversion
 *  when copying data between host and device memory.
 *
 * \latexonly  \subsubsection*{Credits} \endlatexonly
 * \htmlonly   <h3>Credits</h3> \endhtmlonly
 *
 * Some models in this simulator is a rewrite of CSIM (2006) and other
 * work (Stiber and Kawasaki (2007?))
 */
#pragma once

#include "AllSynapses.h"

struct AllSpikingSynapsesDeviceProperties;

typedef void (*fpPreSynapsesSpikeHit_t)(const BGSIZE, AllSpikingSynapsesDeviceProperties *);

typedef void (*fpPostSynapsesSpikeHit_t)(const BGSIZE, AllSpikingSynapsesDeviceProperties *);

class AllSpikingSynapses : public AllSynapses {
public:
   AllSpikingSynapses();

   AllSpikingSynapses(const int numNeurons, const int maxSynapses);

   virtual ~AllSpikingSynapses();

   static IAllSynapses *Create() {
      return new AllSpikingSynapses();
   }

   /**
    *  Setup the internal structure of the class (allocate memories and initialize them).
    */
   virtual void setupSynapses();

   /**
    *  Reset time varying state vars and recompute decay.
    *
    *  @param  iSyn     Index of the synapse to set.
    *  @param  deltaT   Inner simulation step duration
    */
   virtual void resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT);
   /// Load member variables from configuration file. Registered to OperationManager as Operation::op::loadParameters
   virtual void loadParameters();
   /**
    *  Prints out all parameters to logging file.
    *  Registered to OperationManager as Operation::printParameters
    */
   virtual void printParameters() const;

   /**
    *  Create a Synapse and connect it to the model.
    *
    *  @param  iSyn        Index of the synapse to set.
    *  @param  srcNeuron   Coordinates of the source Neuron.
    *  @param  destNeuron  Coordinates of the destination Neuron.
    *  @param  sumPoint    Summation point address.
    *  @param  deltaT      Inner simulation step duration.
    *  @param  type        Type of the Synapse to create.
    */
   virtual void createSynapse(const BGSIZE iSyn, int srcNeuron, int destNeuron, BGFLOAT *sumPoint, const BGFLOAT deltaT,
                              synapseType type);

   /**
    *  Check if the back propagation (notify a spike event to the pre neuron)
    *  is allowed in the synapse class.
    *
    *  @return true if the back propagation is allowed.
    */
   virtual bool allowBackPropagation();

   /**
    *  Prints SynapsesProps data to console.
    */
   virtual void printSynapsesProps() const;

protected:
   /**
    *  Setup the internal structure of the class (allocate memories and initialize them).
    *
    *  @param  numNeurons   Total number of neurons in the network.
    *  @param  maxSynapses  Maximum number of synapses per neuron.
    */
   virtual void setupSynapses(const int numNeurons, const int maxSynapses);

   /**
    *  Initializes the queues for the Synapse.
    *
    *  @param  iSyn   index of the synapse to set.
    */
   virtual void initSpikeQueue(const BGSIZE iSyn);

   /**
    *  Updates the decay if the synapse selected.
    *
    *  @param  iSyn    Index of the synapse to set.
    *  @param  deltaT  Inner simulation step duration
    *  @return true is success.
    */
   bool updateDecay(const BGSIZE iSyn, const BGFLOAT deltaT);

   /**
    *  Sets the data for Synapse to input's data.
    *
    *  @param  input  istream to read from.
    *  @param  iSyn   Index of the synapse to set.
    */
   virtual void readSynapse(istream &input, const BGSIZE iSyn);

   /**
    *  Write the synapse data to the stream.
    *
    *  @param  output  stream to print out to.
    *  @param  iSyn    Index of the synapse to print out.
    */
   virtual void writeSynapse(ostream &output, const BGSIZE iSyn) const;

#if defined(USE_GPU)
   public:
       /**
        *  Allocate GPU memories to store all synapses' states,
        *  and copy them from host to GPU memory.
        *
        *  @param  allSynapsesDevice  GPU address of the allSynapses struct on device memory.
        */
       virtual void allocSynapseDeviceStruct( void** allSynapsesDevice );

       /**
        *  Allocate GPU memories to store all synapses' states,
        *  and copy them from host to GPU memory.
        *
        *  @param  allSynapsesDevice     GPU address of the allSynapses struct on device memory.
        *  @param  numNeurons            Number of neurons.
        *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
        */
       virtual void allocSynapseDeviceStruct( void** allSynapsesDevice, int numNeurons, int maxSynapsesPerNeuron );

       /**
        *  Delete GPU memories.
        *
        *  @param  allSynapsesDevice  GPU address of the allSynapses struct on device memory.
        */
       virtual void deleteSynapseDeviceStruct( void* allSynapsesDevice );

       /**
        *  Copy all synapses' data from host to device.
        *
        *  @param  allSynapsesDevice  GPU address of the allSynapses struct on device memory.
        */
       virtual void copySynapseHostToDevice( void* allSynapsesDevice );

       /**
        *  Copy all synapses' data from host to device.
        *
        *  @param  allSynapsesDevice     GPU address of the allSynapses struct on device memory.
        *  @param  numNeurons            Number of neurons.
        *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
        */
       virtual void copySynapseHostToDevice( void* allSynapsesDevice, int numNeurons, int maxSynapsesPerNeuron );
       /**
        *  Copy all synapses' data from device to host.
        *
        *  @param  allSynapsesDevice  GPU address of the allSynapses struct on device memory.
        */
       virtual void copySynapseDeviceToHost( void* allSynapsesDevice );

       /**
        *  Get synapse_counts in AllSynapses struct on device memory.
        *
        *  @param  allSynapsesDevice  GPU address of the allSynapses struct on device memory.
        */
       virtual void copyDeviceSynapseCountsToHost( void* allSynapsesDevice );

       /**
        *  Get summationCoord and in_use in AllSynapses struct on device memory.
        *
        *  @param  allSynapsesDevice  GPU address of the allSynapses struct on device memory.
        */
       virtual void copyDeviceSynapseSumIdxToHost( void* allSynapsesDevice );

       /**
        *  Advance all the Synapses in the simulation.
        *  Update the state of all synapses for a time step.
        *
        *  @param  allSynapsesDevice      GPU address of the allSynapses struct on device memory.
        *  @param  allNeuronsDevice       GPU address of the allNeurons struct on device memory.
        *  @param  synapseIndexMapDevice  GPU address of the SynapseIndexMap on device memory.
        */
       virtual void advanceSynapses( void* allSynapsesDevice, void* allNeuronsDevice, void* synapseIndexMapDevice );

       /**
        *  Set some parameters used for advanceSynapsesDevice.
        *  Currently we set a member variable: m_fpChangePSR_h.
        */
       virtual void setAdvanceSynapsesDeviceParams( );

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
       virtual void setSynapseClassID( );

       /**
        *  Prints GPU SynapsesProps data.
        *
        *  @param  allSynapsesDeviceProps   GPU address of the corresponding SynapsesDeviceProperties struct on device memory.
        */
       virtual void printGPUSynapsesProps( void* allSynapsesDeviceProps ) const;

   protected:
       /**
        *  Allocate GPU memories to store all synapses' states,
        *  and copy them from host to GPU memory.
        *  (Helper function of allocSynapseDeviceStruct)
        *
        *  @param  allSynapsesDevice     GPU address of the allSynapses struct on device memory.
        *  @param  numNeurons            Number of neurons.
        *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
        */
       void allocDeviceStruct( AllSpikingSynapsesDeviceProperties &allSynapsesDevice, int numNeurons, int maxSynapsesPerNeuron );

       /**
        *  Delete GPU memories.
        *  (Helper function of deleteSynapseDeviceStruct)
        *
        *  @param  allSynapsesDevice  GPU address of the allSynapses struct on device memory.
        */
       void deleteDeviceStruct( AllSpikingSynapsesDeviceProperties& allSynapsesDevice );

       /**
        *  Copy all synapses' data from host to device.
        *  (Helper function of copySynapseHostToDevice)
        *
        *  @param  allSynapsesDevice     GPU address of the allSynapses struct on device memory.
        *  @param  numNeurons            Number of neurons.
        *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
        */
       void copyHostToDevice( void* allSynapsesDevice, AllSpikingSynapsesDeviceProperties& allSynapsesDeviceProps, int numNeurons, int maxSynapsesPerNeuron );

       /**
        *  Copy all synapses' data from device to host.
        *  (Helper function of copySynapseDeviceToHost)
        *
        *  @param  allSynapsesDevice     GPU address of the allSynapses struct on device memory.
        *  @param  numNeurons            Number of neurons.
        *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
        */
       void copyDeviceToHost( AllSpikingSynapsesDeviceProperties& allSynapsesDevice);
#else  // !defined(USE_GPU)
public:
   /**
    *  Advance one specific Synapse.
    *
    *  @param  iSyn      Index of the Synapse to connect to.
    *  @param  neurons   The Neuron list to search from.
    */
   virtual void advanceSynapse(const BGSIZE iSyn, IAllNeurons *neurons);

   /**
    *  Prepares Synapse for a spike hit.
    *
    *  @param  iSyn   Index of the Synapse to update.
    */
   virtual void preSpikeHit(const BGSIZE iSyn);

   /**
    *  Prepares Synapse for a spike hit (for back propagation).
    *
    *  @param  iSyn   Index of the Synapse to update.
    */
   virtual void postSpikeHit(const BGSIZE iSyn);

protected:
   /**
    *  Checks if there is an input spike in the queue.
    *
    *  @param  iSyn   Index of the Synapse to connect to.
    *  @return true if there is an input spike event.
    */
   bool isSpikeQueue(const BGSIZE iSyn);

   /**
    *  Calculate the post synapse response after a spike.
    *
    *  @param  iSyn        Index of the synapse to set.
    *  @param  deltaT      Inner simulation step duration.
    */
   virtual void changePSR(const BGSIZE iSyn, const BGFLOAT deltaT);

#endif

public:

   /**
    *  The decay for the psr.
    */
   BGFLOAT *decay_;

   /**
    *  The synaptic time constant \f$\tau\f$ [units=sec; range=(0,100)].
    */
   BGFLOAT *tau_;

      BGFLOAT tau_II_;
   BGFLOAT tau_IE_;
   BGFLOAT tau_EI_;
   BGFLOAT tau_EE_;
   BGFLOAT delay_II_;
   BGFLOAT delay_IE_;
   BGFLOAT delay_EI_;
   BGFLOAT delay_EE_;

#define BYTES_OF_DELAYQUEUE         ( sizeof(uint32_t) / sizeof(uint8_t) )
#define LENGTH_OF_DELAYQUEUE        ( BYTES_OF_DELAYQUEUE * 8 )

   /**
    *  The synaptic transmission delay, descretized into time steps.
    */
   int *totalDelay_;

   /**
    *  Pointer to the delayed queue.
    */
   uint32_t *delayQueue_;

   /**
    *  The index indicating the current time slot in the delayed queue
    *  Note: This variable is used in GpuSim_struct.cu but I am not sure
    *  if it is actually from a synapse. Will need a little help here. -Aaron
    *  Note: This variable can be GLOBAL VARIABLE, but need to modify the code.
    */
   int *delayIndex_;

   /**
    *  Length of the delayed queue.
    */
   int *delayQueueLength_;

protected:
};

#if defined(USE_GPU)
struct AllSpikingSynapsesDeviceProperties : public AllSynapsesDeviceProperties
{
        /**
         *  The decay for the psr.
         */
        BGFLOAT *decay_;

        /**
         *  The synaptic time constant \f$\tau\f$ [units=sec; range=(0,100)].
         */
        BGFLOAT *tau_;

        /**
         *  The synaptic transmission delay, descretized into time steps.
         */
        int *totalDelay_;

        /**
         *  Pointer to the delayed queue.
         */
        uint32_t *delayQueue_;

        /**
         *  The index indicating the current time slot in the delayed queue
         *  Note: This variable is used in GpuSim_struct.cu but I am not sure 
         *  if it is actually from a synapse. Will need a little help here. -Aaron
         *  Note: This variable can be GLOBAL VARIABLE, but need to modify the code.
         */
        int *delayIndex_;

        /**
         *  Length of the delayed queue.
         */
        int *delayQueueLength_;
};
#endif // defined(USE_GPU)

