/**
 *      @file AllSTDPSynapses.h
 *
 *      @brief A container of all STDP synapse data
 */

/** 
 * @authors Aaron Oziel, Sean Blackbourn
 * 
 * @class AllSTDPSynapses AllSTDPSynapses.h "AllSTDPSynapses.h"
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
 *  For CUDA implementation, we used another structure, AllDSSynapsesDevice, where synapse
 *  parameters are stored in 1D arrays instead of 2D arrays, so that device functions
 *  can access these data less latency. When copying a synapse parameter, P[i][j],
 *  from host to device, it is stored in P[i * max_synapses_per_neuron + j] in 
 *  AllDSSynapsesDevice structure.
 *
 *  The latest implementation uses the identical data struture between host and CUDA;
 *  that is, synapse parameters are stored in a 1D array, so we don't need conversion
 *  when copying data between host and device memory.
 */

/** 
 *  Implements the basic weight update for a time difference \f$Delta =
 *  t_{post}-t_{pre}\f$ with presynaptic spike at time \f$t_{pre}\f$ and
 *  postsynaptic spike at time \f$t_{post}\f$. Then, the weight update is given by
 *  \f$dw =  Apos * exp(-Delta/taupos)\f$ for \f$Delta > 0\f$, and \f$dw =  Aneg *
 *  exp(-Delta/tauneg)\f$ for \f$Delta < 0\f$. (set \f$useFroemkeDanSTDP=0\f$ and
 *  \f$mupos=muneg=0\f$ for this basic update rule).
 *  
 *  It is also possible to use an
 *  extended multiplicative update by changing mupos and muneg. Then \f$dw =
 *  (Wex-W)^{mupos} * Apos * exp(-Delta/taupos)\f$ for \f$Delta > 0\f$ and \f$dw =
 *  W^{mupos} * Aneg * exp(Delta/tauneg)\f$ for \f$Delta < 0\f$. (see Guetig,
 *  Aharonov, Rotter and Sompolinsky (2003). Learning input correlations through
 *  non-linear asymmetric Hebbian plasticity. Journal of Neuroscience 23.
 *  pp.3697-3714.)
 *      
 *  Set \f$useFroemkeDanSTDP=1\f$ (this is the default value) and
 *  use \f$tauspost\f$ and \f$tauspre\f$ for the rule given in Froemke and Dan
 *  (2002). Spike-timing-dependent synaptic modification induced by natural spike
 *  trains. Nature 416 (3/2002). 
 *
 * \latexonly  \subsubsection*{Credits} \endlatexonly
 * \htmlonly   <h3>Credits</h3> \endhtmlonly
 *
 * Some models in this simulator is a rewrite of CSIM (2006) and other
 * work (Stiber and Kawasaki (2007?))
 */

/** 
 *  05/01/2020
 *  Changed the default weight update rule and all formula constants using the 
 *  independent model (a basic STDP model) and multiplicative model in
 *  Froemke and Dan (2002). Spike-timing-dependent synaptic modification induced by natural spike
 *  trains. Nature 416 (3/2002)
 * 
 *  Independent model:
 *  \f$Delta = t_{post}-t_{pre}\f$ with presynaptic spike at time \f$t_{pre}\f$ and
 *  postsynaptic spike at time \f$t_{post}\f$. Then, the weight update is given by
 *  \f$dw =  Apos * exp(-Delta/taupos)\f$ for \f$Delta > 0\f$, and \f$dw =  Aneg *
 *  exp(-Delta/tauneg)\f$ for \f$Delta < 0\f$. dw is the percentage change in synaptic weight.
 *  (set \f$useFroemkeDanSTDP=false\f$ and \f$mupos=muneg=0\f$ for this basic update rule).
 *  
 *  Multiplicative model:
 *  \f$dw = 1.0 + dw * epre * epost\f$ dw is percent change, so adding 1.0 become the scale ratio
 *  \f$W = W * dw\f$ multiply dw (scale ratio) to the current weight to get the new weight
 *  
 *  Note1:This time we don't use useFroemkeDanSTDP (useFroemkeDanSTDP= false) and mupos and muneg (mupos=muneg=0)
 *  Note2:Based on the FroemkeDan paper, the STDP learning rule only applies to excititory synapses, so we
 *  implement it to have only excititory neurons do STDP weight adjustment 
 */

#pragma once

#include "AllSpikingSynapses.h"

struct AllSTDPSynapsesDeviceProperties;

class AllSTDPSynapses : public AllSpikingSynapses {
public:
   AllSTDPSynapses();

   AllSTDPSynapses(const int num_neurons, const int max_synapses);

   virtual ~AllSTDPSynapses();

   static IAllSynapses *Create() { return new AllSTDPSynapses(); }

   /**
    *  Setup the internal structure of the class (allocate memories and initialize them).
    *
    *  @param  sim_info  SimulationInfo class to read information from.
    */
   virtual void setupSynapses();

   /**
    *  Cleanup the class (deallocate memories).
    */
   virtual void cleanupSynapses();

   /**
    *  Reset time varying state vars and recompute decay.
    *
    *  @param  iSyn     Index of the synapse to set.
    *  @param  deltaT   Inner simulation step duration
    */
   virtual void resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT);

   /**
    *  Check if the back propagation (notify a spike event to the pre neuron)
    *  is allowed in the synapse class.
    *
    *  @retrun true if the back propagation is allowed.
    */
   virtual bool allowBackPropagation();

   /**
    *  Prints out all parameters of the neurons to console.
    */
   virtual void printParameters() const;

   /**
    *  Create a Synapse and connect it to the model.
    *
    *  @param  synapses    The synapse list to reference.
    *  @param  iSyn        Index of the synapse to set.
    *  @param  source      Coordinates of the source Neuron.
    *  @param  dest        Coordinates of the destination Neuron.
    *  @param  sum_point   Summation point address.
    *  @param  deltaT      Inner simulation step duration.
    *  @param  type        Type of the Synapse to create.
    */
   virtual void createSynapse(const BGSIZE iSyn, int source_index, int dest_index, BGFLOAT *sp, const BGFLOAT deltaT,
                              synapseType type);

 
protected:
   /**
    *  Setup the internal structure of the class (allocate memories and initialize them).
    *
    *  @param  num_neurons   Total number of neurons in the network.
    *  @param  max_synapses  Maximum number of synapses per neuron.
    */
   virtual void setupSynapses(const int num_neurons, const int max_synapses);



   /**
    *  Initializes the queues for the Synapse.
    *
    *  @param  iSyn   index of the synapse to set.
    */
   virtual void initSpikeQueue(const BGSIZE iSyn);

#if defined(USE_GPU)
   public:
       /**
        *  Allocate GPU memories to store all synapses' states,
        *  and copy them from host to GPU memory.
        *
        *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
        *  @param  sim_info           SimulationInfo to refer from.
        */
       virtual void allocSynapseDeviceStruct( void** allSynapsesDevice, const SimulationInfo *sim_info );

       /**
        *  Allocate GPU memories to store all synapses' states,
        *  and copy them from host to GPU memory.
        *
        *  @param  allSynapsesDevice     Reference to the allSynapses struct on device memory.
        *  @param  num_neurons           Number of neurons.
        *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
        */
       virtual void allocSynapseDeviceStruct( void** allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron );

       /**
        *  Delete GPU memories.
        *
        *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
        *  @param  sim_info           SimulationInfo to refer from.
        */
       virtual void deleteSynapseDeviceStruct( void* allSynapsesDevice );

       /**
        *  Copy all synapses' data from host to device.
        *
        *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
        *  @param  sim_info           SimulationInfo to refer from.
        */
       virtual void copySynapseHostToDevice( void* allSynapsesDevice, const SimulationInfo *sim_info );

       /**
        *  Copy all synapses' data from host to device.
        *
        *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
        *  @param  num_neurons           Number of neurons.
        *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
        */
       virtual void copySynapseHostToDevice( void* allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron );

       /**
        *  Copy all synapses' data from device to host.
        *
        *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
        *  @param  sim_info           SimulationInfo to refer from.
        */
       virtual void copySynapseDeviceToHost( void* allSynapsesDevice, const SimulationInfo *sim_info );

       /**
        *  Advance all the Synapses in the simulation.
        *  Update the state of all synapses for a time step.
        *
        *  @param  allSynapsesDevice      Reference to the allSynapses struct on device memory.
        *  @param  allNeuronsDevice       Reference to the allNeurons struct on device memory.
        *  @param  synapseIndexMapDevice  Reference to the SynapseIndexMap on device memory.
        *  @param  sim_info               SimulationInfo class to read information from.
        */
       virtual void advanceSynapses(void* allSynapsesDevice, void* allNeuronsDevice, void* synapseIndexMapDevice, const SimulationInfo *sim_info);

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
       virtual void setSynapseClassID();

       /**
        *  Prints GPU SynapsesProps data.
        *
        *  @param  allSynapsesDeviceProps   Reference to the corresponding SynapsesDeviceProperties struct on device memory.
        */
       virtual void printGPUSynapsesProps(void* allSynapsesDeviceProps) const;

   protected:
       /**
        *  Allocate GPU memories to store all synapses' states,
        *  and copy them from host to GPU memory.
        *  (Helper function of allocSynapseDeviceStruct)
        *
        *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
        *  @param  num_neurons           Number of neurons.
        *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
        */
       void allocDeviceStruct( AllSTDPSynapsesDeviceProperties &allSynapses, int num_neurons, int maxSynapsesPerNeuron );

       /**
        *  Delete GPU memories.
        *  (Helper function of deleteSynapseDeviceStruct)
        *
        *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
        */
       void deleteDeviceStruct( AllSTDPSynapsesDeviceProperties& allSynapses );

       /**
        *  Copy all synapses' data from host to device.
        *  (Helper function of copySynapseHostToDevice)
        *
        *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
        *  @param  num_neurons           Number of neurons.
        *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
        */
       void copyHostToDevice( void* allSynapsesDevice, AllSTDPSynapsesDeviceProperties& allSynapses, int num_neurons, int maxSynapsesPerNeuron );

       /**
        *  Copy all synapses' data from device to host.
        *  (Helper function of copySynapseDeviceToHost)
        *
        *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
        *  @param  num_neurons           Number of neurons.
        *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
        */
       void copyDeviceToHost( AllSTDPSynapsesDeviceProperties& allSynapses, const SimulationInfo *sim_info );
#else // !defined(USE_GPU)
public:
   /**
    *  Advance one specific Synapse.
    *  Update the state of synapse for a time step
    *
    *  @param  iSyn      Index of the Synapse to connect to.
    *  @param  sim_info  SimulationInfo class to read information from.
    *  @param  neurons   The Neuron list to search from.
    */
   virtual void advanceSynapse(const BGSIZE iSyn, IAllNeurons *neurons);

   /**
    *  Prepares Synapse for a spike hit (for back propagation).
    *
    *  @param  iSyn   Index of the Synapse to connect to.
    */
   virtual void postSpikeHit(const BGSIZE iSyn);

protected:
   /**
    *  Checks if there is an input spike in the queue (for back propagation).
    *
    *  @param  iSyn   Index of the Synapse to connect to.
    *  @return true if there is an input spike event.
    */
   bool isSpikeQueuePost(const BGSIZE iSyn);

private:
   /**
    *  Adjust synapse weight according to the Spike-timing-dependent synaptic modification
    *  induced by natural spike trains
    *
    *  @param  iSyn        Index of the synapse to set.
    *  @param  delta       Pre/post synaptic spike interval.
    *  @param  epost       Params for the rule given in Froemke and Dan (2002).
    *  @param  epre        Params for the rule given in Froemke and Dan (2002).
    */
   void stdpLearning(const BGSIZE iSyn, double delta, double epost, double epre);

#endif
public:
   /**
    *  The synaptic transmission delay (delay of dendritic backpropagating spike),
    *  descretized into time steps.
    */
   int *totalDelayPost_;

   /**
    *  Pointer to the delayed queue
    */
   uint32_t *delayQueuePost_;

   /**
    *  The index indicating the current time slot in the delayed queue.
    */
   int *delayIndexPost_;

   /**
    *  Length of the delayed queue.
    */
   int *delayQueuePostLength_;

   /**
    *  Used for extended rule by Froemke and Dan. See Froemke and Dan (2002).
    *  Spike-timing-dependent synaptic modification induced by natural spike trains.
    *  Nature 416 (3/2002).
    */
   BGFLOAT *tauspost_;

   /**
    *  sed for extended rule by Froemke and Dan.
    */
   BGFLOAT *tauspre_;

   /**
    *  Timeconstant of exponential decay of positive learning window for STDP.
    */
   BGFLOAT *taupos_;

   /**
    *  Timeconstant of exponential decay of negative learning window for STDP.
    */
   BGFLOAT *tauneg_;

   /**
    *  No learning is performed if \f$|Delta| = |t_{post}-t_{pre}| < STDPgap\f$
    */
   BGFLOAT *STDPgap_;

   /**
    *  The maximal/minimal weight of the synapse [readwrite; units=;]
    */
   BGFLOAT *Wex_;

   /**
    *  Defines the peak of the negative exponential learning window.
    */
   BGFLOAT *Aneg_;

   /**
    *  Defines the peak of the positive exponential learning window.
    */
   BGFLOAT *Apos_;

   /**
    *  Extended multiplicative positive update:
    *  \f$dw = (Wex-W)^{mupos} * Apos * exp(-Delta/taupos)\f$.
    *  Set to 0 for basic update. See Guetig, Aharonov, Rotter and Sompolinsky (2003).
    *  Learning input correlations through non-linear asymmetric Hebbian plasticity.
    *  Journal of Neuroscience 23. pp.3697-3714.
    */
   BGFLOAT *mupos_;

   /**
    *  Extended multiplicative negative update:
    *  \f$dw = W^{mupos} * Aneg * exp(Delta/tauneg)\f$. Set to 0 for basic update.
    */
   BGFLOAT *muneg_;

   /**
    *  True if use the rule given in Froemke and Dan (2002).
    */
   bool *useFroemkeDanSTDP_;
};

#if defined(USE_GPU)
struct AllSTDPSynapsesDeviceProperties : public AllSpikingSynapsesDeviceProperties
{
        /**
         *  The synaptic transmission delay (delay of dendritic backpropagating spike), 
         *  descretized into time steps.
         */
        int *totalDelayPost_;

        /**
         *  Pointer to the delayed queue
         */
        uint32_t *delayQueuePost_;

        /**
         *  The index indicating the current time slot in the delayed queue.
         */
        int *delayIndexPost_;

        /**
         *  Length of the delayed queue.
         */
        int *delayQueuePostLength_;

        /**
         *  Used for extended rule by Froemke and Dan. See Froemke and Dan (2002). 
         *  Spike-timing-dependent synaptic modification induced by natural spike trains. 
         *  Nature 416 (3/2002).
         */
        BGFLOAT *tauspost_;

        /**
         *  sed for extended rule by Froemke and Dan.
         */
        BGFLOAT *tauspre_;

        /**
         *  Timeconstant of exponential decay of positive learning window for STDP.
         */
        BGFLOAT *taupos_;

        /**
         *  Timeconstant of exponential decay of negative learning window for STDP.
         */
        BGFLOAT *tauneg_;

        /**
         *  No learning is performed if \f$|Delta| = |t_{post}-t_{pre}| < STDPgap\f$
         */
        BGFLOAT *STDPgap_;

        /**
         *  The maximal/minimal weight of the synapse [readwrite; units=;]
         */
        BGFLOAT *Wex_;

        /**
         *  Defines the peak of the negative exponential learning window.
         */
        BGFLOAT *Aneg_;

        /**
         *  Defines the peak of the positive exponential learning window.
         */
        BGFLOAT *Apos_;

        /**
         *  Extended multiplicative positive update: 
         *  \f$dw = (Wex-W)^{mupos} * Apos * exp(-Delta/taupos)\f$. 
         *  Set to 0 for basic update. See Guetig, Aharonov, Rotter and Sompolinsky (2003). 
         *  Learning input correlations through non-linear asymmetric Hebbian plasticity. 
         *  Journal of Neuroscience 23. pp.3697-3714.
         */
        BGFLOAT *mupos_;

        /**
         *  Extended multiplicative negative update: 
         *  \f$dw = W^{mupos} * Aneg * exp(Delta/tauneg)\f$. Set to 0 for basic update.
         */
        BGFLOAT *muneg_;
  
        /**
         *  True if use the rule given in Froemke and Dan (2002).
         */
        bool *useFroemkeDanSTDP_;
};
#endif // defined(USE_GPU)
