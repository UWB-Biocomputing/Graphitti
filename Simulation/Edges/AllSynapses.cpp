/**
 * @file AllSynapses.cpp
 * 
 * @ingroup Simulation/Edges
 *
 * @brief
 */

#include "AllSynapses.h"
#include "AllNeurons.h"
#include "OperationManager.h"

AllSynapses::AllSynapses() :
      totalSynapseCount_(0),
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

   fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));
}

AllSynapses::AllSynapses(const int numNeurons, const int maxSynapses) {
   setupSynapses(numNeurons, maxSynapses);
}

AllSynapses::~AllSynapses() {
   BGSIZE maxTotalSynapses = maxSynapsesPerNeuron_ * countNeurons_;

  if (maxTotalSynapses != 0) {
     delete[] destNeuronIndex_;
     delete[] W_;
     delete[] summationPoint_;
     delete[] sourceNeuronIndex_;
     delete[] psr_;
     delete[] type_;
     delete[] inUse_;
     delete[] synapseCounts_;
  }

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

///  Setup the internal structure of the class (allocate memories and initialize them).
void AllSynapses::setupSynapses() {
   setupSynapses(Simulator::getInstance().getTotalNeurons(), Simulator::getInstance().getMaxSynapsesPerNeuron());
}

///  Setup the internal structure of the class (allocate memories and initialize them).
///
///  @param  numNeurons   Total number of neurons in the network.
///  @param  max_synapses  Maximum number of synapses per neuron.
void AllSynapses::setupSynapses(const int numNeurons, const int maxSynapses) {
   BGSIZE maxTotalSynapses = maxSynapses * numNeurons;

   maxSynapsesPerNeuron_ = maxSynapses;
   totalSynapseCount_ = 0;
   countNeurons_ = numNeurons;

   if (maxTotalSynapses != 0) {
      destNeuronIndex_ = new int[maxTotalSynapses];
      W_ = new BGFLOAT[maxTotalSynapses];
      summationPoint_ = new BGFLOAT *[maxTotalSynapses];
      sourceNeuronIndex_ = new int[maxTotalSynapses];
      psr_ = new BGFLOAT[maxTotalSynapses];
      type_ = new synapseType[maxTotalSynapses];
      inUse_ = new bool[maxTotalSynapses];
      synapseCounts_ = new BGSIZE[numNeurons];

      for (BGSIZE i = 0; i < maxTotalSynapses; i++) {
         summationPoint_[i] = NULL;
         inUse_[i] = false;
         W_[i] = 0;
      }

      for (int i = 0; i < numNeurons; i++) {
         synapseCounts_[i] = 0;
      }
   }
}

/// Load member variables from configuration file.
/// Registered to OperationManager as Operation::op::loadParameters
void AllSynapses::loadParameters() {
   // Nothing to load from configuration file besides SynapseType in the current implementation.
}

///  Prints out all parameters to logging file.
///  Registered to OperationManager as Operation::printParameters
void AllSynapses::printParameters() const {
   LOG4CPLUS_DEBUG(fileLogger_, "\nEDGES PARAMETERS" << endl
    << "\t---AllSynapses Parameters---" << endl
    << "\tTotal synapse counts: " << totalSynapseCount_ << endl
    << "\tMax synapses per neuron: " << maxSynapsesPerNeuron_ << endl
    << "\tNeuron count: " << countNeurons_ << endl << endl);
}

///  Reset time varying state vars and recompute decay.
///
///  @param  iSyn     Index of the synapse to set.
///  @param  deltaT   Inner simulation step duration
void AllSynapses::resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT) {
   psr_[iSyn] = 0.0;
}

///  Sets the data for Synapse to input's data.
///
///  @param  input  istream to read from.
///  @param  iSyn   Index of the synapse to set.
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

///  Write the synapse data to the stream.
///
///  @param  output  stream to print out to.
///  @param  iSyn    Index of the synapse to print out.
void AllSynapses::writeSynapse(ostream &output, const BGSIZE iSyn) const {
   output << sourceNeuronIndex_[iSyn] << ends;
   output << destNeuronIndex_[iSyn] << ends;
   output << W_[iSyn] << ends;
   output << psr_[iSyn] << ends;
   output << type_[iSyn] << ends;
   output << inUse_[iSyn] << ends;
}

///  Create a synapse index map.
SynapseIndexMap *AllSynapses::createSynapseIndexMap() {
   int neuronCount = Simulator::getInstance().getTotalNeurons();
   int totalSynapseCount = 0;

   // count the total synapses
   for (int i = 0; i < neuronCount; i++) {
      assert(static_cast<int>(synapseCounts_[i]) < Simulator::getInstance().getMaxSynapsesPerNeuron());
      totalSynapseCount += synapseCounts_[i];
   }

   DEBUG (cout << "totalSynapseCount: " << totalSynapseCount << endl;)

   if (totalSynapseCount == 0) {
      return NULL;
   }

   // allocate memories for forward map
   vector<BGSIZE> *rgSynapseSynapseIndexMap = new vector<BGSIZE>[neuronCount];

   BGSIZE syn_i = 0;
   int numInUse = 0;

   // create synapse forward map & active synapse map
   SynapseIndexMap *synapseIndexMap = new SynapseIndexMap(neuronCount, totalSynapseCount);
   for (int i = 0; i < neuronCount; i++) {
      BGSIZE synapse_count = 0;
      synapseIndexMap->incomingSynapseBegin_[i] = numInUse;
      for (int j = 0; j < Simulator::getInstance().getMaxSynapsesPerNeuron(); j++, syn_i++) {
         if (inUse_[syn_i] == true) {
            int idx = sourceNeuronIndex_[syn_i];
            rgSynapseSynapseIndexMap[idx].push_back(syn_i);

            synapseIndexMap->incomingSynapseIndexMap_[numInUse] = syn_i;
            numInUse++;
            synapse_count++;
         }
      }
      assert(synapse_count == this->synapseCounts_[i]);
      synapseIndexMap->incomingSynapseCount_[i] = synapse_count;
   }

   assert(totalSynapseCount == numInUse);
   this->totalSynapseCount_ = totalSynapseCount;

   syn_i = 0;
   for (int i = 0; i < neuronCount; i++) {
      synapseIndexMap->outgoingSynapseBegin_[i] = syn_i;
      synapseIndexMap->outgoingSynapseCount_[i] = rgSynapseSynapseIndexMap[i].size();

      for (BGSIZE j = 0; j < rgSynapseSynapseIndexMap[i].size(); j++, syn_i++) {
         synapseIndexMap->outgoingSynapseIndexMap_[syn_i] = rgSynapseSynapseIndexMap[i][j];
      }
   }

   // delete memories
   delete[] rgSynapseSynapseIndexMap;

   return synapseIndexMap;
}
   
///  Returns an appropriate synapseType object for the given integer.
///
///  @param  typeOrdinal    Integer that correspond with a synapseType.
///  @return the SynapseType that corresponds with the given integer.
synapseType AllSynapses::synapseOrdinalToType(const int typeOrdinal) {
   switch (typeOrdinal) {
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

///  Advance all the Synapses in the simulation.
///
///  @param  neurons           The Neuron list to search from.
///  @param  synapseIndexMap   Pointer to SynapseIndexMap structure.
void AllSynapses::advanceSynapses(IAllNeurons *neurons, SynapseIndexMap *synapseIndexMap) {
   for (BGSIZE i = 0; i < totalSynapseCount_; i++) {
      BGSIZE iSyn = synapseIndexMap->incomingSynapseIndexMap_[i];
      advanceSynapse(iSyn, neurons);
   }
}

///  Remove a synapse from the network.
///
///  @param  neuronIndex    Index of a neuron to remove from.
///  @param  iSyn           Index of a synapse to remove.
void AllSynapses::eraseSynapse(const int neuronIndex, const BGSIZE iSyn) {
   synapseCounts_[neuronIndex]--;
   inUse_[iSyn] = false;
   summationPoint_[iSyn] = NULL;
   W_[iSyn] = 0;
}

#endif // !defined(USE_GPU)

///  Adds a Synapse to the model, connecting two Neurons.
///
///  @param  iSyn        Index of the synapse to be added.
///  @param  type        The type of the Synapse to add.
///  @param  srcNeuron  The Neuron that sends to this Synapse.
///  @param  destNeuron The Neuron that receives from the Synapse.
///  @param  sumPoint   Summation point address.
///  @param  deltaT      Inner simulation step duration
void
AllSynapses::addSynapse(BGSIZE &iSyn, synapseType type, const int srcNeuron, const int destNeuron, BGFLOAT *sumPoint,
                        const BGFLOAT deltaT) {
   if (synapseCounts_[destNeuron] >= maxSynapsesPerNeuron_) {
      LOG4CPLUS_FATAL(fileLogger_, "Neuron : " << destNeuron << " ran out of space for new synapses.");
      throw runtime_error("Neuron : " + destNeuron + string(" ran out of space for new synapses."));
   }

   // add it to the list
   BGSIZE synapseIndex;
   for (synapseIndex = 0; synapseIndex < maxSynapsesPerNeuron_; synapseIndex++) {
      iSyn = maxSynapsesPerNeuron_ * destNeuron + synapseIndex;
      if (!inUse_[iSyn]) {
         break;
      }
   }

   synapseCounts_[destNeuron]++;

   // create a synapse
   createSynapse(iSyn, srcNeuron, destNeuron, sumPoint, deltaT, type);
}

///  Get the sign of the synapseType.
///
///  @param    type    synapseType I to I, I to E, E to I, or E to E
///  @return   1 or -1, or 0 if error
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

///  Prints SynapsesProps data to console.
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

   cout << "totalSynapseCount:" << totalSynapseCount_ << endl;
   cout << "maxSynapsesPerNeuron:" << maxSynapsesPerNeuron_ << endl;
   cout << "count_neurons:" << countNeurons_ << endl;
}





