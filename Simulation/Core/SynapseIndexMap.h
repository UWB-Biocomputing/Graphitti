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

#include <vector>

#include "BGTypes.h"

using namespace std;

struct SynapseIndexMap {
    /// Vector of the outgoing synapse index map.
    vector<BGSIZE> outgoingSynapseIndexMap_;

    /// The beginning index of the outgoing spiking synapse vector of each neuron.
    /// Indexed by a source neuron index.
    vector<BGSIZE> outgoingSynapseBegin_;

    /// The vector of number of outgoing synapses of each neuron.
    /// Indexed by a source neuron index.
    vector<BGSIZE> outgoingSynapseCount_;

    /// Vector of the incoming synapse index map.
    vector<BGSIZE> incomingSynapseIndexMap_;

    /// The beginning index of the incoming spiking synapse vector of each neuron.
    /// Indexed by a destination neuron index.
    vector<BGSIZE> incomingSynapseBegin_;

    /// The vector of number of incoming synapses of each neuron.
    /// Indexed by a destination neuron index.
    vector<BGSIZE> incomingSynapseCount_;

    //vector<BGSIZE> outgoingMapBegin_;

    SynapseIndexMap() : numOfNeurons_(0), numOfSynapses_(0) {};

    SynapseIndexMap(int neuronCount, int synapseCount) : numOfNeurons_(neuronCount), numOfSynapses_(synapseCount) {
        outgoingSynapseIndexMap_.resize(synapseCount);
        outgoingSynapseBegin_.resize(neuronCount);
        outgoingSynapseCount_.resize(neuronCount);

        //outgoingMapBegin_.resize(synapseCount);

        incomingSynapseIndexMap_.resize(synapseCount);
        incomingSynapseBegin_.resize(neuronCount);
        incomingSynapseCount_.resize(neuronCount);
    };

private:
    /// Number of total neurons.
    BGSIZE numOfNeurons_;

    /// Number of total active synapses.
    BGSIZE numOfSynapses_;
};

