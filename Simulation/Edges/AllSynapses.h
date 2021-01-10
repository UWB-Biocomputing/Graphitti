/**
 *      @file AllSynapses.h
 *
 *      @brief A container of all synapse data
 */

/** 
 * @authors Aaron Oziel, Sean Blackbourn
 * 
 * @class AllSynapses AllSynapses.h "AllSynapses.h"
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

#include <log4cplus/loggingmacros.h>

#include "Global.h"
#include "Core/Simulator.h"
#include "IAllSynapses.h"

/**
 * cereal
 */
#include <ThirdParty/cereal/types/vector.hpp>
#include <vector>

#ifdef _WIN32
typedef unsigned _int8 uint8_t;
#endif

class IAllNeurons;

class AllSynapses : public IAllSynapses {
public:
   AllSynapses();

   AllSynapses(const int numNeurons, const int maxSynapses);

   virtual ~AllSynapses();

   /**
    *  Setup the internal structure of the class (allocate memories and initialize them).
    */
   virtual void setupSynapses();

   /**
    * Load member variables from configuration file.
    * Registered to OperationManager as Operation::op::loadParameters
    */
   virtual void loadParameters();

   /**
    *  Prints out all parameters to logging file.
    *  Registered to OperationManager as Operation::printParameters
    */
   virtual void printParameters() const;

   /**
    *  Reset time varying state vars and recompute decay.
    *
    *  @param  iSyn     Index of the synapse to set.
    *  @param  deltaT   Inner simulation step duration
    */
   virtual void resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT);

   /**
    *  Adds a Synapse to the model, connecting two Neurons.
    *
    *  @param  iSyn        Index of the synapse to be added.
    *  @param  type        The type of the Synapse to add.
    *  @param  srcNeuron   The Neuron that sends to this Synapse.
    *  @param  destNeuron  The Neuron that receives from the Synapse.
    *  @param  sumPoint    Summation point address.
    *  @param  deltaT      Inner simulation step duration
    */
   virtual void
   addSynapse(BGSIZE &iSyn, synapseType type, const int srcNeuron, const int destNeuron, BGFLOAT *sumPoint,
              const BGFLOAT deltaT);

   /**
    *  Create a Synapse and connect it to the model.
    *
    *  @param  iSyn        Index of the synapse to set.
    *  @param  source      Coordinates of the source Neuron.
    *  @param  dest        Coordinates of the destination Neuron.
    *  @param  sumPoint    Summation point address.
    *  @param  deltaT      Inner simulation step duration.
    *  @param  type        Type of the Synapse to create.
    */
   virtual void createSynapse(const BGSIZE iSyn, int srcNeuron, int destNeuron, BGFLOAT *sumPoint, const BGFLOAT deltaT,
                              synapseType type) = 0;

   /**
    *  Create a synapse index map and returns it .
    *
    * @return the created SynapseIndexMap
    */
   virtual SynapseIndexMap *createSynapseIndexMap();

   /**
    *  Get the sign of the synapseType.
    *
    *  @param    type    synapseType I to I, I to E, E to I, or E to E
    *  @return   1 or -1, or 0 if error
    */
   int synSign(const synapseType type);


   /**
    *  Cereal serialization method
    *  (Serializes synapse weights, source neurons, and destination neurons)
    */
   template<class Archive>
   void save(Archive &archive) const;

   /**
    *  Cereal deserialization method
    *  (Deserializes synapse weights, source neurons, and destination neurons)
    */
   template<class Archive>
   void load(Archive &archive);

protected:
   /**
    *  Setup the internal structure of the class (allocate memories and initialize them).
    *
    *  @param  numNeurons   Total number of neurons in the network.
    *  @param  maxSynapses  Maximum number of synapses per neuron.
    */
   virtual void setupSynapses(const int numNeurons, const int maxSynapses);

   /**
    *  Returns an appropriate synapseType object for the given integer.
    *
    *  @param  typeOrdinal    Integer that correspond with a synapseType.
    *  @return the SynapseType that corresponds with the given integer.
    */
   synapseType synapseOrdinalToType(const int typeOrdinal);

   /// Loggers used to print to using log4cplus logging macros, prints to Results/Debug/logging.txt
   log4cplus::Logger fileLogger_;
   log4cplus::Logger synapseLogger_;

#if !defined(USE_GPU)
public:
   /**
    *  Advance all the Synapses in the simulation.
    *  Update the state of all synapses for a time step.
    *
    *  @param  neurons   The Neuron list to search from.
    *  @param  synapseIndexMap   Pointer to SynapseIndexMap structure.
    */
   virtual void advanceSynapses(IAllNeurons *neurons, SynapseIndexMap *synapseIndexMap);

   /**
    *  Remove a synapse from the network.
    *
    *  @param  neuronIndex   Index of a neuron to remove from.
    *  @param  iSyn           Index of a synapse to remove.
    */
   virtual void eraseSynapse(const int neuronIndex, const BGSIZE iSyn);

#endif // !defined(USE_GPU)
public:
   // The factor to adjust overlapping area to synapse weight.
   static constexpr BGFLOAT SYNAPSE_STRENGTH_ADJUSTMENT = 1.0e-8;

   /**
    *  The location of the synapse.
    */
   int *sourceNeuronIndex_;

   /**
    *  The coordinates of the summation point.
    */
   int *destNeuronIndex_;

   /**
    *   The weight (scaling factor, strength, maximal amplitude) of the synapse.
    */
   BGFLOAT *W_;

   /**
    *  This synapse's summation point's address.
    */
   BGFLOAT **summationPoint_;

   /**
     *  Synapse type
     */
   synapseType *type_;

   /**
    *  The post-synaptic response is the result of whatever computation
    *  is going on in the synapse.
    */
   BGFLOAT *psr_;

   /**
     *  The boolean value indicating the entry in the array is in use.
     */
   bool *inUse_;

   /**
    *  The number of (incoming) synapses for each neuron.
    *  Note: Likely under a different name in GpuSim_struct, see synapse_count. -Aaron
    */
   BGSIZE *synapseCounts_;

   /**
    *  The total number of active synapses.
    */
   BGSIZE totalSynapseCount_;

   /**
     *  The maximum number of synapses for each neurons.
     */
   BGSIZE maxSynapsesPerNeuron_;

   /**
    *  The number of neurons
    *  Aaron: Is this even supposed to be here?!
    *  Usage: Used by destructor
    */
   int countNeurons_;
};

#if defined(USE_GPU)
struct AllSynapsesDeviceProperties
{
        /**
         *  The location of the synapse.
         */
        int *sourceNeuronIndex_;

        /** 
         *  The coordinates of the summation point.
         */
        int *destNeuronIndex_;

        /**
         *   The weight (scaling factor, strength, maximal amplitude) of the synapse.
         */
         BGFLOAT *W_;

       /**
         *  Synapse type
         */
        synapseType *type_;

        /**
         *  The post-synaptic response is the result of whatever computation
         *  is going on in the synapse.
         */
        BGFLOAT *psr_;

       /**
         *  The boolean value indicating the entry in the array is in use.
         */
        bool *inUse_;

        /**
         *  The number of synapses for each neuron.
         *  Note: Likely under a different name in GpuSim_struct, see synapse_count. -Aaron
         */
        BGSIZE *synapseCounts_;

        /**
         *  The total number of active synapses.
         */
        BGSIZE totalSynapseCount_;

       /**
         *  The maximum number of synapses for each neurons.
         */
        BGSIZE maxSynapsesPerNeuron_;

        /**
         *  The number of neurons
         *  Aaron: Is this even supposed to be here?!
         *  Usage: Used by destructor
         */
        int countNeurons_;
};
#endif // defined(USE_GPU)

/**
 *  Cereal serialization method
 *  (Serializes synapse weights, source neurons, and destination neurons)
 */
template<class Archive>
void AllSynapses::save(Archive &archive) const {
   // uses vector to save synapse weights, source neurons, and destination neurons
   vector<BGFLOAT> WVector;
   vector<int> sourceNeuronLayoutIndexVector;
   vector<int> destNeuronLayoutIndexVector;

   for (int i = 0; i < maxSynapsesPerNeuron_ * countNeurons_; i++) {
      WVector.push_back(W_[i]);
      sourceNeuronLayoutIndexVector.push_back(sourceNeuronIndex_[i]);
      destNeuronLayoutIndexVector.push_back(destNeuronIndex_[i]);
   }

   // serialization
   archive(WVector, sourceNeuronLayoutIndexVector, destNeuronLayoutIndexVector);
}

/**
 *  Cereal deserialization method
 *  (Deserializes synapse weights, source neurons, and destination neurons)
 */
template<class Archive>
void AllSynapses::load(Archive &archive) {
   // uses vectors to load synapse weights, source neurons, and destination neurons
   vector<BGFLOAT> WVector;
   vector<int> sourceNeuronLayoutIndexVector;
   vector<int> destNeuronLayoutIndexVector;

   // deserializing data to these vectors
   archive(WVector, sourceNeuronLayoutIndexVector, destNeuronLayoutIndexVector);

   // check to see if serialized data sizes matches object sizes
   if (WVector.size() != maxSynapsesPerNeuron_ * countNeurons_) {
      cerr
            << "Failed deserializing synapse weights, source neurons, and/or destination neurons. Please verify maxSynapsesPerNeuron and count_neurons data members in AllSynapses class."
            << endl;
      throw cereal::Exception("Deserialization Error");
   }

   // assigns serialized data to objects
   for (int i = 0; i < maxSynapsesPerNeuron_ * countNeurons_; i++) {
      W_[i] = WVector[i];
      sourceNeuronIndex_[i] = sourceNeuronLayoutIndexVector[i];
      destNeuronIndex_[i] = destNeuronLayoutIndexVector[i];
   }
}