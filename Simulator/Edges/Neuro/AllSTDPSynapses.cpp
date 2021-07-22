/**
 * @file AllSTDPSynapses.cpp
 * 
 * @ingroup Simulator/Edges
 *
 * @brief A container of all STDP synapse data
 */

#include "AllSTDPSynapses.h"
#include "AllVertices.h"
#include "AllSpikingNeurons.h"
#include "ParameterManager.h"

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
   defaultSTDPgap_=0;
   tauspost_I_=0;
   tauspre_I_=0;
   tauspost_E_=0;
   tauspre_E_=0;
   taupos_I_=0;
   tauneg_I_=0;
   taupos_E_=0;
   tauneg_E_=0;
   Wex_I_=0;
   Wex_E_=0;
   Aneg_I_ = 0;
   Aneg_E_ = 0;
   Apos_I_ = 0;
   Apos_E_ = 0;
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
   }
}

///  Initializes the queues for the Synapse.
///
///  @param  iEdg   index of the synapse to set.
void AllSTDPSynapses::initSpikeQueue(const BGSIZE iEdg) {
   AllSpikingSynapses::initSpikeQueue(iEdg);

   int &total_delay = totalDelayPost_[iEdg];
   uint32_t &delayQueue = delayQueuePost_[iEdg];
   int &delayIdx = delayIndexPost_[iEdg];
   int &ldelayQueue = delayQueuePostLength_[iEdg];

   uint32_t size = total_delay / (sizeof(uint8_t) * 8) + 1;
   assert(size <= BYTES_OF_DELAYQUEUE);
   delayQueue = 0;
   delayIdx = 0;
   ldelayQueue = LENGTH_OF_DELAYQUEUE;
}

///  Loads out all parameters from the config file.
///  Registered to OperationManager as Operation::loadParameters
void AllSTDPSynapses::loadParameters() {
   AllSpikingSynapses::loadParameters();
   ParameterManager::getInstance().getBGFloatByXpath("//STDPgap/text()", defaultSTDPgap_);
   ParameterManager::getInstance().getBGFloatByXpath("//tauspost/i/text()", tauspost_I_);
   ParameterManager::getInstance().getBGFloatByXpath("//tauspost/e/text()", tauspost_E_);
   ParameterManager::getInstance().getBGFloatByXpath("//tauspre/i/text()", tauspre_I_);
   ParameterManager::getInstance().getBGFloatByXpath("//tauspre/e/text()", tauspre_E_);
   ParameterManager::getInstance().getBGFloatByXpath("//taupos/i/text()", taupos_I_);
   ParameterManager::getInstance().getBGFloatByXpath("//taupos/e/text()", taupos_E_);
   ParameterManager::getInstance().getBGFloatByXpath("//tauneg/i/text()", tauneg_I_);
   ParameterManager::getInstance().getBGFloatByXpath("//tauneg/e/text()", tauneg_E_);
   ParameterManager::getInstance().getBGFloatByXpath("//Wex/i/text()", Wex_I_);
   ParameterManager::getInstance().getBGFloatByXpath("//Wex/e/text()", Wex_E_);
   ParameterManager::getInstance().getBGFloatByXpath("//Aneg/i/text()", Aneg_I_);
   ParameterManager::getInstance().getBGFloatByXpath("//Aneg/e/text()", Aneg_E_);
   ParameterManager::getInstance().getBGFloatByXpath("//Apos/i/text()", Apos_I_);
   ParameterManager::getInstance().getBGFloatByXpath("//Apos/e/text()", Apos_E_);
}

///  Prints out all parameters to logging file.
///  Registered to OperationManager as Operation::printParameters
void AllSTDPSynapses::printParameters() const {
   AllSpikingSynapses::printParameters();
   
   LOG4CPLUS_DEBUG(edgeLogger_, "\n\t---AllSTDPSynapses Parameters---" << endl
                   << "\tEdges type: AllSTDPSynapses" << endl << endl
                   
                   <<"\tSTDP gap" << defaultSTDPgap_ << endl
                   << "\n\tTauspost value: [" << " I: " << tauspost_I_ << ", " << " E: " << tauspost_E_  << "]"<< endl
                   << "\n\tTauspre value: [" << " I: " << tauspre_I_ << ", " << " E: " << tauspre_E_ << "]"<< endl
                   << "\n\tTaupos value: [" << " I: " << taupos_I_ << ", " << " E: " << taupos_E_  << "]"<< endl
                   << "\n\tTau negvalue: [" << " I: " << tauneg_I_ << ", " << " E: " << tauneg_E_  << "]"<< endl
                   << "\n\tWex value: [" << " I: " << Wex_I_ << ", " << " E: " << Wex_E_  << "]"<< endl
                   << "\n\tAneg value: [" << " I: " << Aneg_I_ << ", " << " E: " << Aneg_E_  << "]"<< endl
                   << "\n\tApos value: [" << " I: " << Apos_I_ << ", " << " E: " << Apos_E_  << "]"<< endl
                   );
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
   //Apos_[iEdg] = 0.005;
   //Aneg_[iEdg] = -(1.05*0.005);
   Apos_[iEdg] = Apos_E_;
   Aneg_[iEdg] = Aneg_E_;
   STDPgap_[iEdg] = defaultSTDPgap_;
   tauspost_[iEdg] = tauspost_E_;
   tauspre_[iEdg] = tauspre_E_;
   taupos_[iEdg] = taupos_E_;
   tauneg_[iEdg] = tauneg_E_;
   Wex_[iEdg] = Wex_E_ ;// this is based on overlap of 2 neurons' radii (r=4) of outgrowth, scale it by SYNAPSE_STRENGTH_ADJUSTMENT.
   mupos_[iEdg] = 0;
   muneg_[iEdg] = 0;
}

#if !defined(USE_GPU)

///  Advance one specific Synapse.
///
///  @param  iEdg      Index of the Synapse to connect to.
///  @param  neurons   The Neuron list to search from.
void AllSTDPSynapses::advanceEdge(const BGSIZE iEdg, AllVertices *neurons) {
   // If the synapse is inhibitory or its weight is zero, update synapse state using AllSpikingSynapses::advanceEdge method
   //LOG4CPLUS_DEBUG(edgeLogger_, "iEdg : " << iEdg );
   
   BGFLOAT &W = W_[iEdg];
   /*
   if (W <= 0.0) {
      AllSpikingSynapses::advanceEdge(iEdg, neurons);
      return;
   }
    */

   BGFLOAT &decay = decay_[iEdg];
   BGFLOAT &psr = psr_[iEdg];
   BGFLOAT &summationPoint = *(summationPoint_[iEdg]);

   // is an input in the queue?
   bool fPre = isSpikeQueue(iEdg);
   bool fPost = isSpikeQueuePost(iEdg);

   if (fPre || fPost) {
      const BGFLOAT taupos = taupos_[iEdg];
      const BGFLOAT tauneg = tauneg_[iEdg];
      const int total_delay = totalDelay_[iEdg];

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
         
         epre = 1.0;
         epost = 1.0;
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
            /*
             LOG4CPLUS_DEBUG(fileLogger_,"\nAllSTDPSynapses::advanceSynapse: fPre" << endl
             << "\tiEdg: " << iEdg << endl
             << "\tidxPre: " << idxPre << endl
             << "\tidxPost: " << idxPost << endl
             << "\tspikeHistory: " << spikeHistory << endl
             << "\tepre: " << epre << endl
             << "\tepost: " << epost << endl
             << "\tg_simulationStep: " << g_simulationStep << endl
             << "\tdelta: " << delta << endl << endl);
             */
            
            if (delta <= -3.0 * tauneg)
               break;
            
            stdpLearning(iEdg, delta, epost, epre, idxPre, idxPost);
            --offIndex;
         }

         changePSR(iEdg, deltaT);
      }

      if (fPost) {   // postSpikeHit
                     // spikeCount points to the next available position of spike_history,
                     // so the getSpikeHistory w/offset = -2 will return the spike time
                     // just one before the last spike.
         spikeHistory = spNeurons->getSpikeHistory(idxPost, -2);
         epost = 1.0;
         epre=1;
         
         
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
            /*
             LOG4CPLUS_DEBUG(fileLogger_,"\nAllSTDPSynapses::advanceSynapse: fPost" << endl
             << "\tiEdg: " << iEdg << endl
             << "\tidxPre: " << idxPre << endl
             << "\tidxPost: " << idxPost << endl
             << "\tspikeHistory: " << spikeHistory << endl
             << "\tg_simulationStep: " << g_simulationStep << endl
             << "\tepre: " << epre << endl
             << "\tepost: " << epost << endl
             << "\tdelta: " << delta << endl << endl);
             */
            if (delta >= 3.0 * taupos)
               break;
            
            stdpLearning(iEdg, delta, epost, epre, idxPre, idxPost);
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


BGFLOAT AllSTDPSynapses::synapticWeightModification(const BGSIZE iEdg, BGFLOAT synapticWeight, double delta)
{
   BGFLOAT STDPgap = STDPgap_[iEdg];
   BGFLOAT muneg = muneg_[iEdg];
   BGFLOAT mupos = mupos_[iEdg];
   BGFLOAT tauneg = tauneg_[iEdg];
   BGFLOAT taupos = taupos_[iEdg];
   BGFLOAT Aneg = Aneg_[iEdg];
   BGFLOAT Apos = Apos_[iEdg];
   BGFLOAT Wex = Wex_[iEdg];
   BGFLOAT& W = W_[iEdg];
   edgeType type = type_[iEdg];
   BGFLOAT dw = 0;
   BGFLOAT oldW = W;
   
   // BGFLOAT modDelta = fabs(delta);
   
   if (delta < -STDPgap) {
      // depression
      
      dw = pow(fabs(W) / Wex, muneg) * Aneg * exp(delta / tauneg);  // normalize
   } else if (delta > STDPgap) {
      // potentiation
      dw = pow(fabs(Wex - fabs(W)) / Wex, mupos) * Apos * exp(-delta / taupos); // normalize
   }
   
   return dw;
}


///  Adjust synapse weight according to the Spike-timing-dependent synaptic modification 
///  induced by natural spike trains
///
///  @param  iEdg        Index of the synapse to set.
///  @param  delta       Pre/post synaptic spike interval.
///  @param  epost       Params for the rule given in Froemke and Dan (2002).
///  @param  epre        Params for the rule given in Froemke and Dan (2002).
///  @param srcVertex Index of source neuron
///  @param destVertex Index of destination neuron
void AllSTDPSynapses::stdpLearning(const BGSIZE iEdg, double delta, double epost,
                                   double epre, int srcVertex, int destVertex) {
   BGFLOAT STDPgap = STDPgap_[iEdg];
   BGFLOAT muneg = muneg_[iEdg];
   BGFLOAT mupos = mupos_[iEdg];
   BGFLOAT tauneg = tauneg_[iEdg];
   BGFLOAT taupos = taupos_[iEdg];
   BGFLOAT Aneg = Aneg_[iEdg];
   BGFLOAT Apos = Apos_[iEdg];
   BGFLOAT Wex = Wex_[iEdg];
   BGFLOAT& W = W_[iEdg];
   edgeType type = type_[iEdg];
   BGFLOAT oldW=W;
   // BGFLOAT modDelta = fabs(delta);

   if (delta <= fabs(STDPgap)) {
      return;
   }

   // dw is the fractional change in synaptic strength; add 1.0 to become the scaling ratio
   //dw = 1.0 + dw * epre * epost;
   BGFLOAT dw = 1.0 + synapticWeightModification(iEdg, W, delta);   
   
   // if scaling ratio is less than zero, set it to zero so this synapse, its
   // strength is always zero
   // TODO: Where is the code for this?
   
   // current weight multiplies dw (scaling ratio) to generate new weight
   if(dw != 0.0)
      W *= dw;
   
   // if new weight is bigger than Wex (maximum allowed weight), then set it to Wex
   if (fabs(W) > Wex) {
      W = edgSign(type) * Wex;
   }
   /*
    LOG4CPLUS_DEBUG(edgeLogger_, endl <<
    "iEdg value " << iEdg
    << "; source:" << srcVertex
    << "; dest:" << destVertex
    << "; delta:" << delta
    << "; oldW:" << oldW
    << " ;W:" << W
    << endl);
    
    */
}

///  Checks if there is an input spike in the queue (for back propagation).
///
///  @param  iEdg   Index of the Synapse to connect to.
///  @return true if there is an input spike event.
bool AllSTDPSynapses::isSpikeQueuePost(const BGSIZE iEdg) {
   uint32_t &delayQueue = delayQueuePost_[iEdg];
   int &delayIdx = delayIndexPost_[iEdg];
   int &ldelayQueue = delayQueuePostLength_[iEdg];

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
   uint32_t &delay_queue = delayQueuePost_[iEdg];
   int &delayIdx = delayIndexPost_[iEdg];
   int &ldelayQueue = delayQueuePostLength_[iEdg];
   int &total_delay = totalDelayPost_[iEdg];

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
         cout << " muneg_: " << muneg_[i] << endl;
      }
   }
}
