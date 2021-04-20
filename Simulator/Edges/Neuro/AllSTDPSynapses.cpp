/**
 * @file AllSTDPSynapses.cpp
 * 
 * @ingroup Simulator/Edges
 *
 * @brief A container of all STDP synapse data
 */

#include "AllSTDPSynapses.h"
#include "IAllVertices.h"
#include "AllSpikingNeurons.h"

AllSTDPSynapses::AllSTDPSynapses() : AllSpikingSynapses() {
   totalDelayPost_ = nullptr;
   delayQueuePost_ = nullptr;
   delayIndexPost_ = nullptr;
   delayQueuePostLength_ = nullptr;
   tauspost_ = nullptr;
   tauspre_ = nullptr;
   taupos_ = nullptr;
   tauneg_ = nullptr;
   STDPgap_ = nullptr;
   Wex_ = nullptr;
   Aneg_ = nullptr;
   Apos_ = nullptr;
   mupos_ = nullptr;
   muneg_ = nullptr;
   useFroemkeDanSTDP_ = nullptr;
}

AllSTDPSynapses::AllSTDPSynapses(const int numVertices, const int maxEdges) :
      AllSpikingSynapses(numVertices, maxEdges) {
   setupEdges(numVertices, maxEdges);
}

AllSTDPSynapses::~AllSTDPSynapses() {
   BGSIZE maxTotalSynapses = maxEdgesPerVertex_ * countVertices_;

   if (maxTotalSynapses != 0) {
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

   totalDelayPost_ = nullptr;
   delayQueuePost_ = nullptr;
   delayIndexPost_ = nullptr;
   delayQueuePostLength_ = nullptr;
   tauspost_ = nullptr;
   tauspre_ = nullptr;
   taupos_ = nullptr;
   tauneg_ = nullptr;
   STDPgap_ = nullptr;
   Wex_ = nullptr;
   Aneg_ = nullptr;
   Apos_ = nullptr;
   mupos_ = nullptr;
   muneg_ = nullptr;
   useFroemkeDanSTDP_ = nullptr;
}

///  Setup the internal structure of the class (allocate memories and initialize them).
void AllSTDPSynapses::setupEdges() {
   setupEdges(Simulator::getInstance().getTotalVertices(), Simulator::getInstance().getMaxEdgesPerVertex());
}

///  Setup the internal structure of the class (allocate memories and initialize them).
///
///  @param  numVertices   Total number of vertices in the network.
///  @param  maxEdges  Maximum number of synapses per neuron.
void AllSTDPSynapses::setupEdges(const int numVertices, const int maxEdges) {
   AllSpikingSynapses::setupEdges(numVertices, maxEdges);

   BGSIZE maxTotalSynapses = maxEdges * numVertices;

   if (maxTotalSynapses != 0) {
      totalDelayPost_ = new int[maxTotalSynapses];
      delayQueuePost_ = new uint32_t[maxTotalSynapses];
      delayIndexPost_ = new int[maxTotalSynapses];
      delayQueuePostLength_ = new int[maxTotalSynapses];
      tauspost_ = new BGFLOAT[maxTotalSynapses];
      tauspre_ = new BGFLOAT[maxTotalSynapses];
      taupos_ = new BGFLOAT[maxTotalSynapses];
      tauneg_ = new BGFLOAT[maxTotalSynapses];
      STDPgap_ = new BGFLOAT[maxTotalSynapses];
      Wex_ = new BGFLOAT[maxTotalSynapses];
      Aneg_ = new BGFLOAT[maxTotalSynapses];
      Apos_ = new BGFLOAT[maxTotalSynapses];
      mupos_ = new BGFLOAT[maxTotalSynapses];
      muneg_ = new BGFLOAT[maxTotalSynapses];
      useFroemkeDanSTDP_ = new bool[maxTotalSynapses];
   }
}

///  Initializes the queues for the Synapse.
///
///  @param  iEdg   index of the synapse to set.
void AllSTDPSynapses::initSpikeQueue(const BGSIZE iEdg) {
   AllSpikingSynapses::initSpikeQueue(iEdg);

   int &total_delay = this->totalDelayPost_[iEdg];
   uint32_t &delayQueue = this->delayQueuePost_[iEdg];
   int &delayIdx = this->delayIndexPost_[iEdg];
   int &ldelayQueue = this->delayQueuePostLength_[iEdg];

   uint32_t size = total_delay / (sizeof(uint8_t) * 8) + 1;
   assert(size <= BYTES_OF_DELAYQUEUE);
   delayQueue = 0;
   delayIdx = 0;
   ldelayQueue = LENGTH_OF_DELAYQUEUE;
}

///  Prints out all parameters to logging file.
///  Registered to OperationManager as Operation::printParameters
void AllSTDPSynapses::printParameters() const {
   AllSpikingSynapses::printParameters();
}

///  Sets the data for Synapse to input's data.
///
///  @param  input  istream to read from.
///  @param  iEdg   Index of the synapse to set.
void AllSTDPSynapses::readEdge(istream &input, const BGSIZE iEdg) {
   AllSpikingSynapses::readEdge(input, iEdg);

   // input.ignore() so input skips over end-of-line characters.
   input >> totalDelayPost_[iEdg];
   input.ignore();
   input >> delayQueuePost_[iEdg];
   input.ignore();
   input >> delayIndexPost_[iEdg];
   input.ignore();
   input >> delayQueuePostLength_[iEdg];
   input.ignore();
   input >> tauspost_[iEdg];
   input.ignore();
   input >> tauspre_[iEdg];
   input.ignore();
   input >> taupos_[iEdg];
   input.ignore();
   input >> tauneg_[iEdg];
   input.ignore();
   input >> STDPgap_[iEdg];
   input.ignore();
   input >> Wex_[iEdg];
   input.ignore();
   input >> Aneg_[iEdg];
   input.ignore();
   input >> Apos_[iEdg];
   input.ignore();
   input >> mupos_[iEdg];
   input.ignore();
   input >> muneg_[iEdg];
   input.ignore();
   input >> useFroemkeDanSTDP_[iEdg];
   input.ignore();
}

///  Write the synapse data to the stream.
///
///  @param  output  stream to print out to.
///  @param  iEdg    Index of the synapse to print out.
void AllSTDPSynapses::writeEdge(ostream &output, const BGSIZE iEdg) const {
   AllSpikingSynapses::writeEdge(output, iEdg);

   output << totalDelayPost_[iEdg] << ends;
   output << delayQueuePost_[iEdg] << ends;
   output << delayIndexPost_[iEdg] << ends;
   output << delayQueuePostLength_[iEdg] << ends;
   output << tauspost_[iEdg] << ends;
   output << tauspre_[iEdg] << ends;
   output << taupos_[iEdg] << ends;
   output << tauneg_[iEdg] << ends;
   output << STDPgap_[iEdg] << ends;
   output << Wex_[iEdg] << ends;
   output << Aneg_[iEdg] << ends;
   output << Apos_[iEdg] << ends;
   output << mupos_[iEdg] << ends;
   output << muneg_[iEdg] << ends;
   output << useFroemkeDanSTDP_[iEdg] << ends;
}

///  Reset time varying state vars and recompute decay.
///
///  @param  iEdg            Index of the synapse to set.
///  @param  deltaT          Inner simulation step duration
void AllSTDPSynapses::resetEdge(const BGSIZE iEdg, const BGFLOAT deltaT) {
   AllSpikingSynapses::resetEdge(iEdg, deltaT);
}

///  Create a Synapse and connect it to the model.
///
///  @param  synapses    The synapse list to reference.
///  @param  iEdg        Index of the synapse to set.
///  @param  srcVertex   Coordinates of the source Neuron.
///  @param  destVertex  Coordinates of the destination Neuron.
///  @param  sumPoint    Summation point address.
///  @param  deltaT      Inner simulation step duration.
///  @param  type        Type of the Synapse to create.
void AllSTDPSynapses::createEdge(const BGSIZE iEdg, int srcVertex, int destVertex, BGFLOAT *sumPoint,
                                    const BGFLOAT deltaT, edgeType type) {

   totalDelayPost_[iEdg] = 0;// Apr 12th 2020 move this line so that when AllSpikingSynapses::createEdge() is called, inside this method the initSpikeQueue() method can be called successfully
   AllSpikingSynapses::createEdge(iEdg, srcVertex, destVertex, sumPoint, deltaT, type);

   // May 1st 2020
   // Use constants from Froemke and Dan (2002).
   // Spike-timing-dependent synaptic modification induced by natural spike trains. Nature 416 (3/2002)
   Apos_[iEdg] = 1.01;
   Aneg_[iEdg] = -0.52;
   STDPgap_[iEdg] = 2e-3;

   tauspost_[iEdg] = 75e-3;
   tauspre_[iEdg] = 34e-3;

   taupos_[iEdg] = 14.8e-3;
   tauneg_[iEdg] = 33.8e-3;

   Wex_[iEdg] = 5.0265e-7; // this is based on overlap of 2 neurons' radii (r=4) of outgrowth, scale it by SYNAPSE_STRENGTH_ADJUSTMENT.

   mupos_[iEdg] = 0;
   muneg_[iEdg] = 0;

   useFroemkeDanSTDP_[iEdg] = false;
}

#if !defined(USE_GPU)

///  Advance one specific Synapse.
///
///  @param  iEdg      Index of the Synapse to connect to.
///  @param  neurons   The Neuron list to search from.
void AllSTDPSynapses::advanceEdge(const BGSIZE iEdg, IAllVertices *neurons) {
   // If the synapse is inhibitory or its weight is zero, update synapse state using AllSpikingSynapses::advanceEdge method
   BGFLOAT &W = this->W_[iEdg];
   if (W <= 0.0) {
      AllSpikingSynapses::advanceEdge(iEdg, neurons);
      return;
   }

   BGFLOAT &decay = this->decay_[iEdg];
   BGFLOAT &psr = this->psr_[iEdg];
   BGFLOAT &summationPoint = *(this->summationPoint_[iEdg]);

   // is an input in the queue?
   bool fPre = isSpikeQueue(iEdg);
   bool fPost = isSpikeQueuePost(iEdg);

   if (fPre || fPost) {
      BGFLOAT &tauspre_ = this->tauspre_[iEdg];
      BGFLOAT &tauspost_ = this->tauspost_[iEdg];
      BGFLOAT &taupos_ = this->taupos_[iEdg];
      BGFLOAT &tauneg_ = this->tauneg_[iEdg];
      int &total_delay = this->totalDelay_[iEdg];
      bool &useFroemkeDanSTDP_ = this->useFroemkeDanSTDP_[iEdg];

      BGFLOAT deltaT = Simulator::getInstance().getDeltaT();
      AllSpikingNeurons *spNeurons = dynamic_cast<AllSpikingNeurons *>(neurons);

      // pre and post neurons index
      int idxPre = sourceVertexIndex_[iEdg];
      int idxPost = destVertexIndex_[iEdg];
      uint64_t spikeHistory, spikeHistory2;
      BGFLOAT delta;
      BGFLOAT epre, epost;

      if (fPre) {   // preSpikeHit
         // spikeCount points to the next available position of spike_history,
         // so the getSpikeHistory w/offset = -2 will return the spike time
         // just one before the last spike.
         spikeHistory = spNeurons->getSpikeHistory(idxPre, -2);
         if (spikeHistory != ULONG_MAX && useFroemkeDanSTDP_) {
            // delta will include the transmission delay
            delta = static_cast<BGFLOAT>(g_simulationStep - spikeHistory) * deltaT;
            epre = 1.0 - exp(-delta / tauspre_);
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
            LOG4CPLUS_DEBUG(fileLogger_,"\nAllSTDPSynapses::advanceEdge: fPre" << endl
                   << "\tiSyn: " << iEdg << endl
                   << "\tidxPre: " << idxPre << endl
                   << "\tidxPost: " << idxPost << endl
                   << "\tspikeHistory: " << spikeHistory << endl
                   << "\tg_simulationStep: " << g_simulationStep << endl
                   << "\tdelta: " << delta << endl << endl);

            if (delta <= -3.0 * tauneg_)
               break;
            if (useFroemkeDanSTDP_) {
               spikeHistory2 = spNeurons->getSpikeHistory(idxPost, offIndex - 1);
               if (spikeHistory2 == ULONG_MAX)
                  break;
               epost = 1.0 - exp(-(static_cast<BGFLOAT>(spikeHistory - spikeHistory2) * deltaT) / tauspost_);
            } else {
               epost = 1.0;
            }
            stdpLearning(iEdg, delta, epost, epre);
            --offIndex;
         }

         changePSR(iEdg, deltaT);
      }

      if (fPost) {   // postSpikeHit
         // spikeCount points to the next available position of spike_history,
         // so the getSpikeHistory w/offset = -2 will return the spike time
         // just one before the last spike.
         spikeHistory = spNeurons->getSpikeHistory(idxPost, -2);
         if (spikeHistory != ULONG_MAX && useFroemkeDanSTDP_) {
            // delta will include the transmission delay
            delta = static_cast<BGFLOAT>(g_simulationStep - spikeHistory) * deltaT;
            epost = 1.0 - exp(-delta / tauspost_);
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
            LOG4CPLUS_DEBUG(fileLogger_,"\nAllSTDPSynapses::advanceEdge: fPost" << endl
                   << "\tiSyn: " << iEdg << endl
                   << "\tidxPre: " << idxPre << endl
                   << "\tidxPost: " << idxPost << endl
                   << "\tspikeHistory: " << spikeHistory << endl
                   << "\tg_simulationStep: " << g_simulationStep << endl
                   << "\tdelta: " << delta << endl << endl);

            if (delta >= 3.0 * taupos_)
               break;
            if (useFroemkeDanSTDP_) {
               spikeHistory2 = spNeurons->getSpikeHistory(idxPre, offIndex - 1);
               if (spikeHistory2 == ULONG_MAX)
                  break;
               epre = 1.0 - exp(-(static_cast<BGFLOAT>(spikeHistory - spikeHistory2) * deltaT) / tauspre_);
            } else {
               epre = 1.0;
            }
            stdpLearning(iEdg, delta, epost, epre);
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

///  Adjust synapse weight according to the Spike-timing-dependent synaptic modification 
///  induced by natural spike trains
///
///  @param  iEdg        Index of the synapse to set.
///  @param  delta       Pre/post synaptic spike interval.
///  @param  epost       Params for the rule given in Froemke and Dan (2002).
///  @param  epre        Params for the rule given in Froemke and Dan (2002).
void AllSTDPSynapses::stdpLearning(const BGSIZE iEdg, double delta, double epost, double epre) {
   BGFLOAT STDPgap_ = this->STDPgap_[iEdg];
   BGFLOAT muneg_ = this->muneg_[iEdg];
   BGFLOAT mupos_ = this->mupos_[iEdg];
   BGFLOAT tauneg_ = this->tauneg_[iEdg];
   BGFLOAT taupos_ = this->taupos_[iEdg];
   BGFLOAT Aneg_ = this->Aneg_[iEdg];
   BGFLOAT Apos_ = this->Apos_[iEdg];
   BGFLOAT Wex_ = this->Wex_[iEdg];
   BGFLOAT &W = this->W_[iEdg];
   edgeType type = this->type_[iEdg];
   BGFLOAT dw;

   if (delta < -STDPgap_) {
      // depression
      dw = pow(fabs(W) / Wex_, muneg_) * Aneg_ * exp(delta / tauneg_);  // normalize
   } else if (delta > STDPgap_) {
      // potentiation
      dw = pow(fabs(Wex_ - fabs(W)) / Wex_, mupos_) * Apos_ * exp(-delta / taupos_); // normalize
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

   // if new weight is bigger than Wex_ (maximum allowed weight), then set it to Wex_
   if (fabs(W) > Wex_) {
      W = edgSign(type) * Wex_;
   }

   LOG4CPLUS_DEBUG(fileLogger_,
         "AllSTDPSynapses::stdpLearning:" << endl
          << "\tiSyn: " << iEdg << endl
          << "\tdelta: " << delta << endl
          << "\tepre: " << epre << endl
          << "\tepost: " << epost << endl
          << "\tdw: " << dw << endl
          << "\tW: " << W << endl << endl);
}

///  Checks if there is an input spike in the queue (for back propagation).
///
///  @param  iEdg   Index of the Synapse to connect to.
///  @return true if there is an input spike event.
bool AllSTDPSynapses::isSpikeQueuePost(const BGSIZE iEdg) {
   uint32_t &delayQueue = this->delayQueuePost_[iEdg];
   int &delayIdx = this->delayIndexPost_[iEdg];
   int &ldelayQueue = this->delayQueuePostLength_[iEdg];

   bool r = delayQueue & (0x1 << delayIdx);
   delayQueue &= ~(0x1 << delayIdx);
   if (++delayIdx >= ldelayQueue) {
      delayIdx = 0;
   }
   return r;
}

///  Prepares Synapse for a spike hit (for back propagation).
///
///  @param  iEdg   Index of the Synapse to connect to.
void AllSTDPSynapses::postSpikeHit(const BGSIZE iEdg) {
   uint32_t &delay_queue = this->delayQueuePost_[iEdg];
   int &delayIdx = this->delayIndexPost_[iEdg];
   int &ldelayQueue = this->delayQueuePostLength_[iEdg];
   int &total_delay = this->totalDelayPost_[iEdg];

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

///  Check if the back propagation (notify a spike event to the pre neuron)
///  is allowed in the synapse class.
///
///  @retrun true if the back propagation is allowed.
bool AllSTDPSynapses::allowBackPropagation() {
   return true;
}

///  Prints SynapsesProps data.
void AllSTDPSynapses::printSynapsesProps() const {
   AllSpikingSynapses::printSynapsesProps();
   for (int i = 0; i < maxEdgesPerVertex_ * countVertices_; i++) {
      if (W_[i] != 0.0) {
         cout << "total_delayPost[" << i << "] = " << totalDelayPost_[i];
         cout << " tauspost_: " << tauspost_[i];
         cout << " tauspre_: " << tauspre_[i];
         cout << " taupos_: " << taupos_[i];
         cout << " tauneg_: " << tauneg_[i];
         cout << " STDPgap_: " << STDPgap_[i];
         cout << " Wex_: " << Wex_[i];
         cout << " Aneg_: " << Aneg_[i];
         cout << " Apos_: " << Apos_[i];
         cout << " mupos_: " << mupos_[i];
         cout << " muneg_: " << muneg_[i];
         cout << " useFroemkeDanSTDP_: " << useFroemkeDanSTDP_[i] << endl;
      }
   }
}
