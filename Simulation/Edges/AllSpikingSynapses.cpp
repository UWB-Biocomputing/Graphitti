#include "AllSpikingSynapses.h"
#include "ParameterManager.h"
#include "OperationManager.h"

AllSpikingSynapses::AllSpikingSynapses() : AllSynapses() {
   decay_ = NULL;
   totalDelay_ = NULL;
   delayQueue_ = NULL;
   delayIndex_ = NULL;
   delayQueueLength_ = NULL;
   tau_ = NULL;
   tau_II_=NULL;
   tau_IE_=NULL;
   tau_EI_=NULL;
   tau_EE_=NULL;
   delay_II_=NULL;
   delay_IE_=NULL;
   delay_EI_=NULL;
   delay_EE_=NULL;
}

AllSpikingSynapses::AllSpikingSynapses(const int num_neurons, const int max_synapses) {
   setupSynapses(num_neurons, max_synapses);
}

AllSpikingSynapses::~AllSpikingSynapses() {
   cleanupSynapses();
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 */
void AllSpikingSynapses::setupSynapses() {
   setupSynapses(Simulator::getInstance().getTotalNeurons(), Simulator::getInstance().getMaxSynapsesPerNeuron());
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  num_neurons   Total number of neurons in the network.
 *  @param  max_synapses  Maximum number of synapses per neuron.
 */
void AllSpikingSynapses::setupSynapses(const int num_neurons, const int max_synapses) {
   AllSynapses::setupSynapses(num_neurons, max_synapses);

   BGSIZE max_total_synapses = max_synapses * num_neurons;

   if (max_total_synapses != 0) {
      decay_ = new BGFLOAT[max_total_synapses];
      totalDelay_ = new int[max_total_synapses];
      delayQueue_ = new uint32_t[max_total_synapses];
      delayIndex_ = new int[max_total_synapses];
      delayQueueLength_ = new int[max_total_synapses];
      tau_ = new BGFLOAT[max_total_synapses];
      tau_II_=0;
      tau_IE_=0;
      tau_EI_=0;
      tau_EE_=0;
      delay_II_=0;
      delay_IE_=0;
      delay_EI_=0;
      delay_EE_=0;
     
   }
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllSpikingSynapses::cleanupSynapses() {
   BGSIZE max_total_synapses = maxSynapsesPerNeuron_ * countNeurons_;

//   if (max_total_synapses != 0) {
//      delete[] decay_;
//      delete[] totalDelay_;
//      delete[] delayQueue_;
//      delete[] delayIndex_;
//      delete[] delayQueueLength_;
//      delete[] tau_;
//   }

   decay_ = NULL;
   totalDelay_ = NULL;
   delayQueue_ = NULL;
   delayIndex_ = NULL;
   delayQueueLength_ = NULL;
   tau_ = NULL;

   AllSynapses::cleanupSynapses();
}

/*
 *  Initializes the queues for the Synapse.
 *
 *  @param  iSyn   index of the synapse to set.
 */
void AllSpikingSynapses::initSpikeQueue(const BGSIZE iSyn) {
   int &total_delay = this->totalDelay_[iSyn];
   uint32_t &delayQueue = this->delayQueue_[iSyn];
   int &delayIdx = this->delayIndex_[iSyn];
   int &ldelayQueue = this->delayQueueLength_[iSyn];

   uint32_t size = total_delay / (sizeof(uint8_t) * 8) + 1;
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

<<<<<<< HEAD
void AllSpikingSynapses::loadParameters() {
   ParameterManager::getInstance().getBGFloatByXpath("//tau/ii/text()", tau_II_);
   ParameterManager::getInstance().getBGFloatByXpath("//tau/ie/text()", tau_IE_);
   ParameterManager::getInstance().getBGFloatByXpath("//tau/ei/text()", tau_EI_);
   ParameterManager::getInstance().getBGFloatByXpath("//tau/ee/text()", tau_EE_);
   ParameterManager::getInstance().getBGFloatByXpath("//delay/ii/text()", delay_II_);
   ParameterManager::getInstance().getBGFloatByXpath("//delay/ie/text()", delay_IE_);
   ParameterManager::getInstance().getBGFloatByXpath("//delay/ei/text()", delay_EI_);
   ParameterManager::getInstance().getBGFloatByXpath("//delay/ee/text()", delay_EE_);
=======
/*
 *  Prints out all parameters of the synapses to console.
 */
void AllSpikingSynapses::printParameters() const {
   AllSynapses::printParameters();
>>>>>>> e311573ca2647649ee14566d3a3788b42aa1b6d0
}

/*
 *  Prints out all parameters of the neurons to ostream.
 * 
 *  @param  output  ostream to send output to.
 */
void AllSpikingSynapses::printParameters() const {
   cout << "\tTau values: ["
          << " II: " << tau_II_ << ", " << " IE: " << tau_IE_ << "," << "EI : " << tau_EI_<< "," << " EE: " << tau_EE_ << "]"
          << endl;

    cout << "\tDelay values: ["
          << " II: "<< delay_II_ << ", " << " IE: "<< delay_IE_ << "," << "EI :" << delay_EI_<< "," << " EE: "<< delay_EE_ << "]"
          << endl;
}


/*
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
void AllSpikingSynapses::createSynapse(const BGSIZE iSyn, int source_index, int dest_index, BGFLOAT *sum_point,
                                       const BGFLOAT deltaT, synapseType type) {
   BGFLOAT delay;
   BGFLOAT tau;

   inUse_[iSyn] = true;
   summationPoint_[iSyn] = sum_point;
   destNeuronIndex_[iSyn] = dest_index;
   sourceNeuronIndex_[iSyn] = source_index;
   W_[iSyn] = synSign(type) * 10.0e-9;
   this->type_[iSyn] = type;
   tau_[iSyn] = DEFAULT_tau;

   
   switch (type) {
      case II:
         tau = tau_II_;
         delay = delay_II_;
         break;
      case IE:
         tau = tau_IE_;
         delay = delay_IE_;
         break;
      case EI:
         tau = tau_EI_;
         delay = delay_EI_;
         break;
      case EE:
         tau = tau_EE_;
         delay = delay_EE_;
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
   uint32_t &delay_queue = this->delayQueue_[iSyn];
   int &delayIdx = this->delayIndex_[iSyn];
   int &ldelayQueue = this->delayQueueLength_[iSyn];
   int &total_delay = this->totalDelay_[iSyn];

   // Add to spike queue

   // calculate index where to insert the spike into delayQueue
   int idx = delayIdx + total_delay;
   if (idx >= ldelayQueue) {
      idx -= ldelayQueue;
   }

   // set a spike
   assert(!(delay_queue & (0x1 << idx)));
   delay_queue |= (0x1 << idx);
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
 *  @param  sim_info  SimulationInfo class to read information from.
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
 *  @retrun true if the back propagation is allowed.
 */
bool AllSpikingSynapses::allowBackPropagation() {
   return false;
}
