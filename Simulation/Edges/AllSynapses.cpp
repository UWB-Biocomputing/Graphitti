#include "AllSynapses.h"
#include "AllNeurons.h"
#include "OperationManager.h"

AllSynapses::AllSynapses() :
      totalSynapseCounts_(0),
      maxSynapsesPerNeuron_(0),
      countNeurons_(0) {
   destNeuronIndex_ = NULL;
   W_ = NULL;
   summationPoint_ = NULL;
   sourceNeuronIndex_ = NULL;
   psr_ = NULL;
   type_ = NULL;
   inUse_ = NULL;
   synapseCounts_ = NULL;

   // Register loadParameters function as a loadParameters operation in the OperationManager
   function<void()> loadParametersFunc = std::bind(&IAllSynapses::loadParameters, this);
   OperationManager::getInstance().registerOperation(Operations::op::loadParameters, loadParametersFunc);

   // Register printParameters function as a printParameters operation in the OperationManager
   function<void()> printParametersFunc = bind(&IAllSynapses::printParameters, this);
   OperationManager::getInstance().registerOperation(Operations::printParameters, printParametersFunc);
}

AllSynapses::AllSynapses(const int num_neurons, const int max_synapses) {
   setupSynapses(num_neurons, max_synapses);
}

AllSynapses::~AllSynapses() {
   cleanupSynapses();
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 */
void AllSynapses::setupSynapses() {
   setupSynapses(Simulator::getInstance().getTotalNeurons(), Simulator::getInstance().getMaxSynapsesPerNeuron());
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  num_neurons   Total number of neurons in the network.
 *  @param  max_synapses  Maximum number of synapses per neuron.
 */
void AllSynapses::setupSynapses(const int num_neurons, const int max_synapses) {
   BGSIZE max_total_synapses = max_synapses * num_neurons;

   maxSynapsesPerNeuron_ = max_synapses;
   totalSynapseCounts_ = 0;
   countNeurons_ = num_neurons;

   if (max_total_synapses != 0) {
      destNeuronIndex_ = new int[max_total_synapses];
      W_ = new BGFLOAT[max_total_synapses];
      summationPoint_ = new BGFLOAT *[max_total_synapses];
      sourceNeuronIndex_ = new int[max_total_synapses];
      psr_ = new BGFLOAT[max_total_synapses];
      type_ = new synapseType[max_total_synapses];
      inUse_ = new bool[max_total_synapses];
      synapseCounts_ = new BGSIZE[num_neurons];

      for (BGSIZE i = 0; i < max_total_synapses; i++) {
         summationPoint_[i] = NULL;
         inUse_[i] = false;
         W_[i] = 0;
      }

      for (int i = 0; i < num_neurons; i++) {
         synapseCounts_[i] = 0;
      }
   }
}

/**
 * Load member variables from configuration file.
 * Registered to OperationManager as Operation::op::loadParameters
 */
void AllSynapses::loadParameters() {
   // Nothing to load from configuration file besides SynapseType in the current implementation.
}

/**
 *  Prints out all parameters of the neurons to console.
 */
void AllSynapses::printParameters() const {
   cout << "EDGES PARAMETERS" << endl;
   cout << "\t*AllSynapses Parameters*" << endl;
   cout << "\tTotal synapse counts: " << totalSynapseCounts_ << endl;
   cout << "\tMax synapses per neuron: " << maxSynapsesPerNeuron_ << endl;
   cout << "\tNeuron count: " << countNeurons_ << endl << endl;
}


/*
 *  Cleanup the class (deallocate memories).
 */
void AllSynapses::cleanupSynapses() {
   BGSIZE max_total_synapses = maxSynapsesPerNeuron_ * countNeurons_;

//   if (max_total_synapses != 0) {
//      delete[] destNeuronIndex_;
//      delete[] W_;
//      delete[] summationPoint_;
//      delete[] sourceNeuronIndex_;
//      delete[] psr_;
//      delete[] type_;
//      delete[] inUse_;
//      delete[] synapseCounts_;
//   }

   destNeuronIndex_ = NULL;
   W_ = NULL;
   summationPoint_ = NULL;
   sourceNeuronIndex_ = NULL;
   psr_ = NULL;
   type_ = NULL;
   inUse_ = NULL;
   synapseCounts_ = NULL;

   countNeurons_ = 0;
   maxSynapsesPerNeuron_ = 0;
}

/*
 *  Reset time varying state vars and recompute decay.
 *
 *  @param  iSyn     Index of the synapse to set.
 *  @param  deltaT   Inner simulation step duration
 */
void AllSynapses::resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT) {
   psr_[iSyn] = 0.0;
}

/*
 *  Sets the data for Synapse to input's data.
 *
 *  @param  input  istream to read from.
 *  @param  iSyn   Index of the synapse to set.
 */
void AllSynapses::readSynapse(istream &input, const BGSIZE iSyn) {
   int synapse_type(0);

   // input.ignore() so input skips over end-of-line characters.
   input >> sourceNeuronIndex_[iSyn];
   input.ignore();
   input >> destNeuronIndex_[iSyn];
   input.ignore();
   input >> W_[iSyn];
   input.ignore();
   input >> psr_[iSyn];
   input.ignore();
   input >> synapse_type;
   input.ignore();
   input >> inUse_[iSyn];
   input.ignore();

   type_[iSyn] = synapseOrdinalToType(synapse_type);
}

/*
 *  Write the synapse data to the stream.
 *
 *  @param  output  stream to print out to.
 *  @param  iSyn    Index of the synapse to print out.
 */
void AllSynapses::writeSynapse(ostream &output, const BGSIZE iSyn) const {
   output << sourceNeuronIndex_[iSyn] << ends;
   output << destNeuronIndex_[iSyn] << ends;
   output << W_[iSyn] << ends;
   output << psr_[iSyn] << ends;
   output << type_[iSyn] << ends;
   output << inUse_[iSyn] << ends;
}

/*
 *  Create a synapse index map.
 *
 *  @param  synapseIndexMap   Reference to the pointer to SynapseIndexMap structure.
 *  @param  sim_info          Pointer to the simulation information.
 */
void AllSynapses::createSynapseImap(SynapseIndexMap *synapseIndexMap) {
   int neuron_count = Simulator::getInstance().getTotalNeurons();
   int total_synapse_counts = 0;

   // count the total synapses
   for (int i = 0; i < neuron_count; i++) {
      assert(static_cast<int>(synapseCounts_[i]) < Simulator::getInstance().getMaxSynapsesPerNeuron());
      total_synapse_counts += synapseCounts_[i];
   }

   DEBUG (cout << "total_synapse_counts: " << total_synapse_counts << endl;)

   if (total_synapse_counts == 0) {
      return;
   }

   // allocate memories for forward map
   vector<BGSIZE> *rgSynapseSynapseIndexMap = new vector<BGSIZE>[neuron_count];

   if (synapseIndexMap != NULL) {
      delete synapseIndexMap;
      synapseIndexMap = NULL;
   }

   BGSIZE syn_i = 0;
   int n_inUse = 0;

   // create synapse forward map & active synapse map
   synapseIndexMap = new SynapseIndexMap(neuron_count, total_synapse_counts);
   for (int i = 0; i < neuron_count; i++) {
      BGSIZE synapse_count = 0;
      synapseIndexMap->incomingSynapseBegin_[i] = n_inUse;
      for (int j = 0; j < Simulator::getInstance().getMaxSynapsesPerNeuron(); j++, syn_i++) {
         if (inUse_[syn_i] == true) {
            int idx = sourceNeuronIndex_[syn_i];
            rgSynapseSynapseIndexMap[idx].push_back(syn_i);

            synapseIndexMap->incomingSynapseIndexMap_[n_inUse] = syn_i;
            n_inUse++;
            synapse_count++;
         }
      }
      assert(synapse_count == this->synapseCounts_[i]);
      synapseIndexMap->incomingSynapseCount_[i] = synapse_count;
   }

   assert(total_synapse_counts == n_inUse);
   this->totalSynapseCounts_ = total_synapse_counts;

   syn_i = 0;
   for (int i = 0; i < neuron_count; i++) {
      synapseIndexMap->outgoingSynapseBegin_[i] = syn_i;
      synapseIndexMap->outgoingSynapseCount_[i] = rgSynapseSynapseIndexMap[i].size();

      for (BGSIZE j = 0; j < rgSynapseSynapseIndexMap[i].size(); j++, syn_i++) {
         synapseIndexMap->outgoingSynapseIndexMap_[syn_i] = rgSynapseSynapseIndexMap[i][j];
      }
   }

   // delete memories
   delete[] rgSynapseSynapseIndexMap;
}

/*     
 *  Returns an appropriate synapseType object for the given integer.
 *
 *  @param  type_ordinal    Integer that correspond with a synapseType.
 *  @return the SynapseType that corresponds with the given integer.
 */
synapseType AllSynapses::synapseOrdinalToType(const int type_ordinal) {
   switch (type_ordinal) {
      case 0:
         return II;
      case 1:
         return IE;
      case 2:
         return EI;
      case 3:
         return EE;
      default:
         return STYPE_UNDEF;
   }
}

#if !defined(USE_GPU)

/*
 *  Advance all the Synapses in the simulation.
 *
 *  @param  sim_info          SimulationInfo class to read information from.
 *  @param  neurons           The Neuron list to search from.
 *  @param  synapseIndexMap   Pointer to SynapseIndexMap structure.
 */
void AllSynapses::advanceSynapses(IAllNeurons *neurons, SynapseIndexMap *synapseIndexMap) {
   for (BGSIZE i = 0; i < totalSynapseCounts_; i++) {
      BGSIZE iSyn = synapseIndexMap->incomingSynapseIndexMap_[i];
      advanceSynapse(iSyn, neurons);
   }
}

/*
 *  Remove a synapse from the network.
 *
 *  @param  neuron_index   Index of a neuron to remove from.
 *  @param  iSyn           Index of a synapse to remove.
 */
void AllSynapses::eraseSynapse(const int neuron_index, const BGSIZE iSyn) {
   synapseCounts_[neuron_index]--;
   inUse_[iSyn] = false;
   summationPoint_[iSyn] = NULL;
   W_[iSyn] = 0;
}

#endif // !defined(USE_GPU)

/*
 *  Adds a Synapse to the model, connecting two Neurons.
 *
 *  @param  iSyn        Index of the synapse to be added.
 *  @param  type        The type of the Synapse to add.
 *  @param  src_neuron  The Neuron that sends to this Synapse.
 *  @param  dest_neuron The Neuron that receives from the Synapse.
 *  @param  sum_point   Summation point address.
 *  @param  deltaT      Inner simulation step duration
 */
void
AllSynapses::addSynapse(BGSIZE &iSyn, synapseType type, const int src_neuron, const int dest_neuron, BGFLOAT *sum_point,
                        const BGFLOAT deltaT) {
   if (synapseCounts_[dest_neuron] >= maxSynapsesPerNeuron_) {
      DEBUG (cout << "Neuron : " << dest_neuron << " ran out of space for new synapses." << endl;)
      return; // TODO: ERROR!
   }

   // add it to the list
   BGSIZE synapse_index;
   for (synapse_index = 0; synapse_index < maxSynapsesPerNeuron_; synapse_index++) {
      iSyn = maxSynapsesPerNeuron_ * dest_neuron + synapse_index;
      if (!inUse_[iSyn]) {
         break;
      }
   }

   synapseCounts_[dest_neuron]++;

   // create a synapse
   createSynapse(iSyn, src_neuron, dest_neuron, sum_point, deltaT, type);
}

/*
 *  Get the sign of the synapseType.
 *
 *  @param    type    synapseType I to I, I to E, E to I, or E to E
 *  @return   1 or -1, or 0 if error
 */
int AllSynapses::synSign(const synapseType type) {
   switch (type) {
      case II:
      case IE:
         return -1;
      case EI:
      case EE:
         return 1;
      case STYPE_UNDEF:
         // TODO error.
         return 0;
   }

   return 0;
}

/*
 *  Prints SynapsesProps data.
 */
void AllSynapses::printSynapsesProps() const {
   cout << "This is SynapsesProps data:" << endl;
   for (int i = 0; i < maxSynapsesPerNeuron_ * countNeurons_; i++) {
      if (W_[i] != 0.0) {
         cout << "W[" << i << "] = " << W_[i];
         cout << " sourNeuron: " << sourceNeuronIndex_[i];
         cout << " desNeuron: " << destNeuronIndex_[i];
         cout << " type: " << type_[i];
         cout << " psr: " << psr_[i];
         cout << " in_use:" << inUse_[i];
         if (summationPoint_[i] != nullptr) {
            cout << " summationPoint: is created!" << endl;
         } else {
            cout << " summationPoint: is EMPTY!!!!!" << endl;
         }
      }
   }

   for (int i = 0; i < countNeurons_; i++) {
      cout << "synapse_counts:" << "neuron[" << i << "]" << synapseCounts_[i] << endl;
   }

   cout << "total_synapse_counts:" << totalSynapseCounts_ << endl;
   cout << "maxSynapsesPerNeuron:" << maxSynapsesPerNeuron_ << endl;
   cout << "count_neurons:" << countNeurons_ << endl;
}





