#include "AllSTDPSynapses.h"
#include "IAllNeurons.h"
#include "AllSpikingNeurons.h"

AllSTDPSynapses::AllSTDPSynapses() : AllSpikingSynapses() {
   totalDelayPost_ = NULL;
   delayQueuePost_ = NULL;
   delayIndexPost_ = NULL;
   delayQueuePostLength_ = NULL;
   tauspost_ = NULL;
   tauspre_ = NULL;
   taupos_ = NULL;
   tauneg_ = NULL;
   STDPgap_ = NULL;
   Wex_ = NULL;
   Aneg_ = NULL;
   Apos_ = NULL;
   mupos_ = NULL;
   muneg_ = NULL;
   useFroemkeDanSTDP_ = NULL;
}

AllSTDPSynapses::AllSTDPSynapses(const int num_neurons, const int max_synapses) :
      AllSpikingSynapses(num_neurons, max_synapses) {
   setupSynapses(num_neurons, max_synapses);
}

AllSTDPSynapses::~AllSTDPSynapses() {
   cleanupSynapses();
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 */
void AllSTDPSynapses::setupSynapses() {
   setupSynapses(Simulator::getInstance().getTotalNeurons(), Simulator::getInstance().getMaxSynapsesPerNeuron());
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  num_neurons   Total number of neurons in the network.
 *  @param  max_synapses  Maximum number of synapses per neuron.
 */
void AllSTDPSynapses::setupSynapses(const int num_neurons, const int max_synapses) {
   AllSpikingSynapses::setupSynapses(num_neurons, max_synapses);

   BGSIZE max_total_synapses = max_synapses * num_neurons;

   if (max_total_synapses != 0) {
      totalDelayPost_ = new int[max_total_synapses];
      delayQueuePost_ = new uint32_t[max_total_synapses];
      delayIndexPost_ = new int[max_total_synapses];
      delayQueuePostLength_ = new int[max_total_synapses];
      tauspost_ = new BGFLOAT[max_total_synapses];
      tauspre_ = new BGFLOAT[max_total_synapses];
      taupos_ = new BGFLOAT[max_total_synapses];
      tauneg_ = new BGFLOAT[max_total_synapses];
      STDPgap_ = new BGFLOAT[max_total_synapses];
      Wex_ = new BGFLOAT[max_total_synapses];
      Aneg_ = new BGFLOAT[max_total_synapses];
      Apos_ = new BGFLOAT[max_total_synapses];
      mupos_ = new BGFLOAT[max_total_synapses];
      muneg_ = new BGFLOAT[max_total_synapses];
      useFroemkeDanSTDP_ = new bool[max_total_synapses];
   }
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllSTDPSynapses::cleanupSynapses() {
   BGSIZE max_total_synapses = maxSynapsesPerNeuron_ * countNeurons_;

   if (max_total_synapses != 0) {
      delete[] totalDelayPost_;
      delete[] delayQueuePost_;
      delete[] delayIndexPost_;
      delete[] delayQueuePostLength_;
      delete[] tauspost_;
      delete[] tauspre_;
      delete[] taupos_;
      delete[] tauneg_;
      delete[] STDPgap_;
      delete[] Wex_;
      delete[] Aneg_;
      delete[] Apos_;
      delete[] mupos_;
      delete[] muneg_;
      delete[] useFroemkeDanSTDP_;
   }

   totalDelayPost_ = NULL;
   delayQueuePost_ = NULL;
   delayIndexPost_ = NULL;
   delayQueuePostLength_ = NULL;
   tauspost_ = NULL;
   tauspre_ = NULL;
   taupos_ = NULL;
   tauneg_ = NULL;
   STDPgap_ = NULL;
   Wex_ = NULL;
   Aneg_ = NULL;
   Apos_ = NULL;
   mupos_ = NULL;
   muneg_ = NULL;
   useFroemkeDanSTDP_ = NULL;

   AllSpikingSynapses::cleanupSynapses();
}

/*
 *  Initializes the queues for the Synapse.
 *
 *  @param  iSyn   index of the synapse to set.
 */
void AllSTDPSynapses::initSpikeQueue(const BGSIZE iSyn) {
   AllSpikingSynapses::initSpikeQueue(iSyn);

   int &total_delay = this->totalDelayPost_[iSyn];
   uint32_t &delayQueue = this->delayQueuePost_[iSyn];
   int &delayIdx = this->delayIndexPost_[iSyn];
   int &ldelayQueue = this->delayQueuePostLength_[iSyn];

   uint32_t size = total_delay / (sizeof(uint8_t) * 8) + 1;
   assert(size <= BYTES_OF_DELAYQUEUE);
   delayQueue = 0;
   delayIdx = 0;
   ldelayQueue = LENGTH_OF_DELAYQUEUE;
}

/*
 *  Prints out all parameters of the neurons to console.
 */
void AllSTDPSynapses::printParameters() const {
}


/*
 *  Reset time varying state vars and recompute decay.
 *
 *  @param  iSyn            Index of the synapse to set.
 *  @param  deltaT          Inner simulation step duration
 */
void AllSTDPSynapses::resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT) {
   AllSpikingSynapses::resetSynapse(iSyn, deltaT);
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
void AllSTDPSynapses::createSynapse(const BGSIZE iSyn, int source_index, int dest_index, BGFLOAT *sum_point,
                                    const BGFLOAT deltaT, synapseType type) {

   totalDelayPost_[iSyn] = 0;// Apr 12th 2020 move this line so that when AllSpikingSynapses::createSynapse() is called, inside this method the initSpikeQueue() method can be called successfully
   AllSpikingSynapses::createSynapse(iSyn, source_index, dest_index, sum_point, deltaT, type);

   // May 1st 2020
   // Use constants from Froemke and Dan (2002).
   // Spike-timing-dependent synaptic modification induced by natural spike trains. Nature 416 (3/2002)
   Apos_[iSyn] = 1.01;
   Aneg_[iSyn] = -0.52;
   STDPgap_[iSyn] = 2e-3;

   tauspost_[iSyn] = 75e-3;
   tauspre_[iSyn] = 34e-3;

   taupos_[iSyn] = 14.8e-3;
   tauneg_[iSyn] = 33.8e-3;

   Wex_[iSyn] = 5.0265e-7; // this is based on overlap of 2 neurons' radii (r=4) of outgrowth, scale it by SYNAPSE_STRENGTH_ADJUSTMENT.

   mupos_[iSyn] = 0;
   muneg_[iSyn] = 0;

   useFroemkeDanSTDP_[iSyn] = false;
}

#if !defined(USE_GPU)

/*
 *  Advance one specific Synapse.
 *
 *  @param  iSyn      Index of the Synapse to connect to.
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  neurons   The Neuron list to search from.
 */
void AllSTDPSynapses::advanceSynapse(const BGSIZE iSyn, IAllNeurons *neurons) {
   // If the synapse is inhibitory or its weight is zero, update synapse state using AllSpikingSynapses::advanceSynapse method
   BGFLOAT &W = this->W_[iSyn];
   if (W <= 0.0) {
      AllSpikingSynapses::advanceSynapse(iSyn, neurons);
      return;
   }

   BGFLOAT &decay = this->decay_[iSyn];
   BGFLOAT &psr = this->psr_[iSyn];
   BGFLOAT &summationPoint = *(this->summationPoint_[iSyn]);

   // is an input in the queue?
   bool fPre = isSpikeQueue(iSyn);
   bool fPost = isSpikeQueuePost(iSyn);

   if (fPre || fPost) {
      BGFLOAT &tauspre = this->tauspre_[iSyn];
      BGFLOAT &tauspost = this->tauspost_[iSyn];
      BGFLOAT &taupos = this->taupos_[iSyn];
      BGFLOAT &tauneg = this->tauneg_[iSyn];
      int &total_delay = this->totalDelay_[iSyn];
      bool &useFroemkeDanSTDP = this->useFroemkeDanSTDP_[iSyn];

      BGFLOAT deltaT = Simulator::getInstance().getDeltaT();
      AllSpikingNeurons *spNeurons = dynamic_cast<AllSpikingNeurons *>(neurons);

      // pre and post neurons index
      int idxPre = sourceNeuronIndex_[iSyn];
      int idxPost = destNeuronIndex_[iSyn];
      uint64_t spikeHistory, spikeHistory2;
      BGFLOAT delta;
      BGFLOAT epre, epost;

      if (fPre) {   // preSpikeHit
         // spikeCount points to the next available position of spike_history,
         // so the getSpikeHistory w/offset = -2 will return the spike time
         // just one before the last spike.
         spikeHistory = spNeurons->getSpikeHistory(idxPre, -2);
         if (spikeHistory != ULONG_MAX && useFroemkeDanSTDP) {
            // delta will include the transmission delay
            delta = static_cast<BGFLOAT>(g_simulationStep - spikeHistory) * deltaT;
            epre = 1.0 - exp(-delta / tauspre);
         } else {
            epre = 1.0;
         }

         // call the learning function stdpLearning() for each pair of
         // pre-post spikes
         int offIndex = -1;   // last spike
         while (true) {
            spikeHistory = spNeurons->getSpikeHistory(idxPost, offIndex);
            if (spikeHistory == ULONG_MAX)
               break;
            // delta is the spike interval between pre-post spikes
            // (include pre-synaptic transmission delay)
            delta = -static_cast<BGFLOAT>(g_simulationStep - spikeHistory) * deltaT;
            DEBUG_SYNAPSE(
                  cout << "AllSTDPSynapses::advanceSynapse: fPre" << endl;
                  cout << "          iSyn: " << iSyn << endl;
                  cout << "          idxPre: " << idxPre << endl;
                  cout << "          idxPost: " << idxPost << endl;
                  cout << "          spikeHistory: " << spikeHistory << endl;
                  cout << "          g_simulationStep: " << g_simulationStep << endl;
                  cout << "          delta: " << delta << endl << endl;
            );

            if (delta <= -3.0 * tauneg)
               break;
            if (useFroemkeDanSTDP) {
               spikeHistory2 = spNeurons->getSpikeHistory(idxPost, offIndex - 1);
               if (spikeHistory2 == ULONG_MAX)
                  break;
               epost = 1.0 - exp(-(static_cast<BGFLOAT>(spikeHistory - spikeHistory2) * deltaT) / tauspost);
            } else {
               epost = 1.0;
            }
            stdpLearning(iSyn, delta, epost, epre);
            --offIndex;
         }

         changePSR(iSyn, deltaT);
      }

      if (fPost) {   // postSpikeHit
         // spikeCount points to the next available position of spike_history,
         // so the getSpikeHistory w/offset = -2 will return the spike time
         // just one before the last spike.
         spikeHistory = spNeurons->getSpikeHistory(idxPost, -2);
         if (spikeHistory != ULONG_MAX && useFroemkeDanSTDP) {
            // delta will include the transmission delay
            delta = static_cast<BGFLOAT>(g_simulationStep - spikeHistory) * deltaT;
            epost = 1.0 - exp(-delta / tauspost);
         } else {
            epost = 1.0;
         }

         // call the learning function stdpLearning() for each pair of
         // post-pre spikes
         int offIndex = -1;   // last spike
         while (true) {
            spikeHistory = spNeurons->getSpikeHistory(idxPre, offIndex);
            if (spikeHistory == ULONG_MAX)
               break;

            if (spikeHistory + total_delay > g_simulationStep) {
               --offIndex;
               continue;
            }
            // delta is the spike interval between post-pre spikes
            delta = static_cast<BGFLOAT>(g_simulationStep - spikeHistory - total_delay) * deltaT;
            DEBUG_SYNAPSE(
                  cout << "AllSTDPSynapses::advanceSynapse: fPost" << endl;
                  cout << "          iSyn: " << iSyn << endl;
                  cout << "          idxPre: " << idxPre << endl;
                  cout << "          idxPost: " << idxPost << endl;
                  cout << "          spikeHistory: " << spikeHistory << endl;
                  cout << "          g_simulationStep: " << g_simulationStep << endl;
                  cout << "          delta: " << delta << endl << endl;
            );

            if (delta >= 3.0 * taupos)
               break;
            if (useFroemkeDanSTDP) {
               spikeHistory2 = spNeurons->getSpikeHistory(idxPre, offIndex - 1);
               if (spikeHistory2 == ULONG_MAX)
                  break;
               epre = 1.0 - exp(-(static_cast<BGFLOAT>(spikeHistory - spikeHistory2) * deltaT) / tauspre);
            } else {
               epre = 1.0;
            }
            stdpLearning(iSyn, delta, epost, epre);
            --offIndex;
         }
      }
   }

   // decay the post spike response
   psr *= decay;
   // and apply it to the summation point
#ifdef USE_OMP
#pragma omp atomic
#endif
   summationPoint += psr;
#ifdef USE_OMP
   //PAB: atomic above has implied flush (following statement generates error -- can't be member variable)
   //#pragma omp flush (summationPoint)
#endif

}

/*
 *  Adjust synapse weight according to the Spike-timing-dependent synaptic modification 
 *  induced by natural spike trains
 *
 *  @param  iSyn        Index of the synapse to set.
 *  @param  delta       Pre/post synaptic spike interval.
 *  @param  epost       Params for the rule given in Froemke and Dan (2002).
 *  @param  epre        Params for the rule given in Froemke and Dan (2002).
 */
void AllSTDPSynapses::stdpLearning(const BGSIZE iSyn, double delta, double epost, double epre) {
   BGFLOAT STDPgap = this->STDPgap_[iSyn];
   BGFLOAT muneg = this->muneg_[iSyn];
   BGFLOAT mupos = this->mupos_[iSyn];
   BGFLOAT tauneg = this->tauneg_[iSyn];
   BGFLOAT taupos = this->taupos_[iSyn];
   BGFLOAT Aneg = this->Aneg_[iSyn];
   BGFLOAT Apos = this->Apos_[iSyn];
   BGFLOAT Wex = this->Wex_[iSyn];
   BGFLOAT &W = this->W_[iSyn];
   synapseType type = this->type_[iSyn];
   BGFLOAT dw;

   if (delta < -STDPgap) {
      // depression
      dw = pow(fabs(W) / Wex, muneg) * Aneg * exp(delta / tauneg);  // normalize
   } else if (delta > STDPgap) {
      // potentiation
      dw = pow(fabs(Wex - fabs(W)) / Wex, mupos) * Apos * exp(-delta / taupos); // normalize
   } else {
      return;
   }

   // dw is the percentage change in synaptic strength; add 1.0 to become the scaling ratio
   dw = 1.0 + dw * epre * epost;

   // if scaling ratio is less than zero, set it to zero so this synapse, its strength is always zero
   if (dw < 0) {
      dw = 0;
   }

   // current weight multiplies dw (scaling ratio) to generate new weight
   W *= dw;

   // if new weight is bigger than Wex (maximum allowed weight), then set it to Wex
   if (fabs(W) > Wex) {
      W = synSign(type) * Wex;
   }

   DEBUG_SYNAPSE(
         cout << "AllSTDPSynapses::stdpLearning:" << endl;
         cout << "          iSyn: " << iSyn << endl;
         cout << "          delta: " << delta << endl;
         cout << "          epre: " << epre << endl;
         cout << "          epost: " << epost << endl;
         cout << "          dw: " << dw << endl;
         cout << "          W: " << W << endl << endl;
   );
}

/*
 *  Checks if there is an input spike in the queue (for back propagation).
 *
 *  @param  iSyn   Index of the Synapse to connect to.
 *  @return true if there is an input spike event.
 */
bool AllSTDPSynapses::isSpikeQueuePost(const BGSIZE iSyn) {
   uint32_t &delayQueue = this->delayQueuePost_[iSyn];
   int &delayIdx = this->delayIndexPost_[iSyn];
   int &ldelayQueue = this->delayQueuePostLength_[iSyn];

   bool r = delayQueue & (0x1 << delayIdx);
   delayQueue &= ~(0x1 << delayIdx);
   if (++delayIdx >= ldelayQueue) {
      delayIdx = 0;
   }
   return r;
}

/*
 *  Prepares Synapse for a spike hit (for back propagation).
 *
 *  @param  iSyn   Index of the Synapse to connect to.
 */
void AllSTDPSynapses::postSpikeHit(const BGSIZE iSyn) {
   uint32_t &delay_queue = this->delayQueuePost_[iSyn];
   int &delayIdx = this->delayIndexPost_[iSyn];
   int &ldelayQueue = this->delayQueuePostLength_[iSyn];
   int &total_delay = this->totalDelayPost_[iSyn];

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

#endif // !defined(USE_GPU)

/*
 *  Check if the back propagation (notify a spike event to the pre neuron)
 *  is allowed in the synapse class.
 *
 *  @retrun true if the back propagation is allowed.
 */
bool AllSTDPSynapses::allowBackPropagation() {
   return true;
}

