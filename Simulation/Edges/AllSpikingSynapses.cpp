#include "AllSpikingSynapses.h"

AllSpikingSynapses::AllSpikingSynapses() : AllSynapses() {
   decay_ = NULL;
   totalDelay_ = NULL;
   delayQueue_ = NULL;
   delayIndex_ = NULL;
   delayQueueLength_ = NULL;
   tau_ = NULL;
}

AllSpikingSynapses::AllSpikingSynapses(const int numNeurons, const int maxSynapses) {
   setupSynapses(numNeurons, maxSynapses);
}

AllSpikingSynapses::~AllSpikingSynapses() {
   BGSIZE maxTotalSynapses = maxSynapsesPerNeuron_ * countNeurons_;
  
  if (maxTotalSynapses != 0) {
      delete[] decay_;
      delete[] totalDelay_;
      delete[] delayQueue_;
      delete[] delayIndex_;
      delete[] delayQueueLength_;
      delete[] tau_;
  }

   decay_ = NULL;
   totalDelay_ = NULL;
   delayQueue_ = NULL;
   delayIndex_ = NULL;
   delayQueueLength_ = NULL;
   tau_ = NULL;
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 */
void AllSpikingSynapses::setupSynapses() {
   setupSynapses(Simulator::getInstance().getTotalNeurons(), Simulator::getInstance().getMaxSynapsesPerNeuron());
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  numNeurons   Total number of neurons in the network.
 *  @param  maxSynapses  Maximum number of synapses per neuron.
 */
void AllSpikingSynapses::setupSynapses(const int numNeurons, const int maxSynapses) {
   AllSynapses::setupSynapses(numNeurons, maxSynapses);

   BGSIZE maxTotalSynapses = maxSynapses * numNeurons;

   if (maxTotalSynapses != 0) {
      decay_ = new BGFLOAT[maxTotalSynapses];
      totalDelay_ = new int[maxTotalSynapses];
      delayQueue_ = new uint32_t[maxTotalSynapses];
      delayIndex_ = new int[maxTotalSynapses];
      delayQueueLength_ = new int[maxTotalSynapses];
      tau_ = new BGFLOAT[maxTotalSynapses];
   }
}

/*
 *  Initializes the queues for the Synapse.
 *
 *  @param  iSyn   index of the synapse to set.
 */
void AllSpikingSynapses::initSpikeQueue(const BGSIZE iSyn) {
   int &totalDelay = this->totalDelay_[iSyn];
   uint32_t &delayQueue = this->delayQueue_[iSyn];
   int &delayIdx = this->delayIndex_[iSyn];
   int &ldelayQueue = this->delayQueueLength_[iSyn];

   uint32_t size = totalDelay / (sizeof(uint8_t) * 8) + 1;
   assert(size <= BYTES_OF_DELAYQUEUE);
   delayQueue = 0;
   delayIdx = 0;
   ldelayQueue = LENGTH_OF_DELAYQUEUE;
}

/*
 *  Reset time varying state vars and recompute decay.
 *
 *  @param  iSyn     Index of the synapse to set.
 *  @param  deltaT   Inner simulation step duration
 */
void AllSpikingSynapses::resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT) {
   AllSynapses::resetSynapse(iSyn, deltaT);

   assert(updateDecay(iSyn, deltaT));
}

/**
 *  Prints out all parameters to logging file.
 *  Registered to OperationManager as Operation::printParameters
 */
void AllSpikingSynapses::printParameters() const {
   AllSynapses::printParameters();
}

/*
 *  Sets the data for Synapse to input's data.
 *
 *  @param  input  istream to read from.
 *  @param  iSyn   Index of the synapse to set.
 */
void AllSpikingSynapses::readSynapse(istream &input, const BGSIZE iSyn) {
   AllSynapses::readSynapse(input, iSyn);

   // input.ignore() so input skips over end-of-line characters.
   input >> decay_[iSyn];
   input.ignore();
   input >> totalDelay_[iSyn];
   input.ignore();
   input >> delayQueue_[iSyn];
   input.ignore();
   input >> delayIndex_[iSyn];
   input.ignore();
   input >> delayQueueLength_[iSyn];
   input.ignore();
   input >> tau_[iSyn];
   input.ignore();
}

/*
 *  Write the synapse data to the stream.
 *
 *  @param  output  stream to print out to.
 *  @param  iSyn    Index of the synapse to print out.
 */
void AllSpikingSynapses::writeSynapse(ostream &output, const BGSIZE iSyn) const {
   AllSynapses::writeSynapse(output, iSyn);

   output << decay_[iSyn] << ends;
   output << totalDelay_[iSyn] << ends;
   output << delayQueue_[iSyn] << ends;
   output << delayIndex_[iSyn] << ends;
   output << delayQueueLength_[iSyn] << ends;
   output << tau_[iSyn] << ends;
}

/*
 *  Create a Synapse and connect it to the model.
 *
 *  @param  iSyn        Index of the synapse to set.
 *  @param  srcNeuron   Coordinates of the source Neuron.
 *  @param  destNeuron  Coordinates of the destination Neuron.
 *  @param  sumPoint    Summation point address.
 *  @param  deltaT      Inner simulation step duration.
 *  @param  type        Type of the Synapse to create.
 */
void AllSpikingSynapses::createSynapse(const BGSIZE iSyn, int srcNeuron, int destNeuron, BGFLOAT *sumPoint,
                                       const BGFLOAT deltaT, synapseType type) {
   BGFLOAT delay;

   inUse_[iSyn] = true;
   summationPoint_[iSyn] = sumPoint;
   destNeuronIndex_[iSyn] = destNeuron;
   sourceNeuronIndex_[iSyn] = srcNeuron;
   W_[iSyn] = synSign(type) * 10.0e-9;
   this->type_[iSyn] = type;
   tau_[iSyn] = DEFAULT_tau;

   BGFLOAT tau;
   switch (type) {
      case II:
         tau = 6e-3;
         delay = 0.8e-3;
         break;
      case IE:
         tau = 6e-3;
         delay = 0.8e-3;
         break;
      case EI:
         tau = 3e-3;
         delay = 0.8e-3;
         break;
      case EE:
         tau = 3e-3;
         delay = 1.5e-3;
         break;
      default:
         assert(false);
         break;
   }

   this->tau_[iSyn] = tau;
   this->totalDelay_[iSyn] = static_cast<int>( delay / deltaT ) + 1;

   // initializes the queues for the Synapses
   initSpikeQueue(iSyn);
   // reset time varying state vars and recompute decay
   resetSynapse(iSyn, deltaT);
}

#if !defined(USE_GPU)

/*
 *  Checks if there is an input spike in the queue.
 *
 *  @param  iSyn   Index of the Synapse to connect to.
 *  @return true if there is an input spike event.
 */
bool AllSpikingSynapses::isSpikeQueue(const BGSIZE iSyn) {
   uint32_t &delayQueue = this->delayQueue_[iSyn];
   int &delayIdx = this->delayIndex_[iSyn];
   int &ldelayQueue = this->delayQueueLength_[iSyn];

   bool r = delayQueue & (0x1 << delayIdx);
   delayQueue &= ~(0x1 << delayIdx);
   if (++delayIdx >= ldelayQueue) {
      delayIdx = 0;
   }
   return r;
}

/*
 *  Prepares Synapse for a spike hit.
 *
 *  @param  iSyn   Index of the Synapse to update.
 */
void AllSpikingSynapses::preSpikeHit(const BGSIZE iSyn) {
   uint32_t &delayQueue = this->delayQueue_[iSyn];
   int &delayIdx = this->delayIndex_[iSyn];
   int &ldelayQueue = this->delayQueueLength_[iSyn];
   int &totalDelay = this->totalDelay_[iSyn];

   // Add to spike queue

   // calculate index where to insert the spike into delayQueue
   int idx = delayIdx + totalDelay;
   if (idx >= ldelayQueue) {
      idx -= ldelayQueue;
   }

   // set a spike
   assert(!(delayQueue & (0x1 << idx)));
   delayQueue |= (0x1 << idx);
}

/*
 *  Prepares Synapse for a spike hit (for back propagation).
 *
 *  @param  iSyn   Index of the Synapse to update.
 */
void AllSpikingSynapses::postSpikeHit(const BGSIZE iSyn) {
}

/*
 *  Advance one specific Synapse.
 *
 *  @param  iSyn      Index of the Synapse to connect to.
 *  @param  neurons   The Neuron list to search from.
 */
void AllSpikingSynapses::advanceSynapse(const BGSIZE iSyn, IAllNeurons *neurons) {
   BGFLOAT &decay = this->decay_[iSyn];
   BGFLOAT &psr = this->psr_[iSyn];
   BGFLOAT &summationPoint = *(this->summationPoint_[iSyn]);

   // is an input in the queue?
   if (isSpikeQueue(iSyn)) {
      changePSR(iSyn, Simulator::getInstance().getDeltaT());
   }

   // decay the post spike response
   psr *= decay;
   // and apply it to the summation point
#ifdef USE_OMP
#pragma omp atomic #endif
#endif
   summationPoint += psr;
#ifdef USE_OMP
   //PAB: atomic above has implied flush (following statement generates error -- can't be member variable)
   //#pragma omp flush (summationPoint)
#endif
}

/*
 *  Calculate the post synapse response after a spike.
 *
 *  @param  iSyn        Index of the synapse to set.
 *  @param  deltaT      Inner simulation step duration.
 */
void AllSpikingSynapses::changePSR(const BGSIZE iSyn, const BGFLOAT deltaT) {
   BGFLOAT &psr = this->psr_[iSyn];
   BGFLOAT &W = this->W_[iSyn];
   BGFLOAT &decay = this->decay_[iSyn];

   psr += (W / decay);    // calculate psr
}

#endif //!defined(USE_GPU)

/*
 *  Updates the decay if the synapse selected.
 *
 *  @param  iSyn    Index of the synapse to set.
 *  @param  deltaT  Inner simulation step duration
 */
bool AllSpikingSynapses::updateDecay(const BGSIZE iSyn, const BGFLOAT deltaT) {
   BGFLOAT &tau = this->tau_[iSyn];
   BGFLOAT &decay = this->decay_[iSyn];

   if (tau > 0) {
      decay = exp(-deltaT / tau);
      return true;
   }
   return false;
}

/*
 *  Check if the back propagation (notify a spike event to the pre neuron)
 *  is allowed in the synapse class.
 *
 *  @return true if the back propagation is allowed.
 */
bool AllSpikingSynapses::allowBackPropagation() {
   return false;
}

/*
 *  Prints SynapsesProps data to console.
 */
void AllSpikingSynapses::printSynapsesProps() const {
   AllSynapses::printSynapsesProps();
   for (int i = 0; i < maxSynapsesPerNeuron_ * countNeurons_; i++) {
      if (W_[i] != 0.0) {
         cout << "decay[" << i << "] = " << decay_[i];
         cout << " tau: " << tau_[i];
         cout << " total_delay: " << totalDelay_[i] << endl;
      }
   }
}
