/**
 *  @file SynapseIndexMap.h
 *
 *  @brief A structure maintains outgoing and active synapses list (forward map).
 *
 *  @ingroup Core
 *
 *  @struct SynapseIndexMap SynapseIndexMap.h "SynapseIndexMap.h"
 *
 *  \latexonly  \subsubsection*{Implementation} \endlatexonly
 *  \htmlonly   <h3>Implementation</h3> \endhtmlonly
 *
 *  The structure maintains a list of outgoing synapses (forward map) and active synapses list.
 *
 *  The outgoing synapses list stores all outgoing synapse indexes relevant to a neuron.
 *  Synapse indexes are stored in the synapse forward map (forwardIndex), and
 *  the pointer and length of the neuron i's outgoing synapse indexes are specified
 *  by outgoingSynapse_begin[i] and synapseCount[i] respectively.
 *  The incoming synapses list is used in calcSummationMapDevice() device function to
 *  calculate sum of PSRs for each neuron simultaneously.
 *  The list also used in AllSpikingNeurons::advanceNeurons() function to allow back propagation.
 *
 *  The active synapses list stores all active synapse indexes.
 *  The list is referred in advanceSynapsesDevice() device function.
 *  The list contribute to reduce the number of the device function thread to skip the inactive
 *  synapses.
 *
 *  \latexonly  \subsubsection*{Credits} \endlatexonly
 *  \htmlonly   <h3>Credits</h3> \endhtmlonly
 */

#pragma once

#include "BGTypes.h"

using namespace std;

struct SynapseIndexMap {
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

   SynapseIndexMap() : numOfNeurons_(0), numOfSynapses_(0) {
      outgoingSynapseBegin_ = NULL;
      outgoingSynapseCount_ = NULL;
      incomingSynapseBegin_ = NULL;
      incomingSynapseCount_ = NULL;

      outgoingSynapseIndexMap_ = NULL;
      incomingSynapseIndexMap_ = NULL;
   };

   SynapseIndexMap(int neuronCount, int synapseCount) : numOfNeurons_(neuronCount), numOfSynapses_(synapseCount) {
      if (numOfNeurons_ > 0) {
         outgoingSynapseBegin_ = new BGSIZE[numOfNeurons_];
         outgoingSynapseCount_ = new BGSIZE[numOfNeurons_];
         incomingSynapseBegin_ = new BGSIZE[numOfNeurons_];
         incomingSynapseCount_ = new BGSIZE[numOfNeurons_];
      }

      if (numOfSynapses_ > 0) {
         outgoingSynapseIndexMap_ = new BGSIZE[numOfSynapses_];
         incomingSynapseIndexMap_ = new BGSIZE[numOfSynapses_];
      }
   };

   ~SynapseIndexMap() {
      if (numOfNeurons_ > 0) {
         delete[] outgoingSynapseBegin_;
         delete[] outgoingSynapseCount_;
         delete[] incomingSynapseBegin_;
         delete[] incomingSynapseCount_;
      }
      if (numOfSynapses_ > 0) {
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

