
#include "AllSTDPSynapses.h"
#include "IAllNeurons.h"
#include "AllSpikingNeurons.h"
#include "ParameterManager.h"
#include "OperationManager.h"

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
   STDPgap=0;
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

AllSTDPSynapses::AllSTDPSynapses(const int numNeurons, const int maxSynapses) :
      AllSpikingSynapses(numNeurons, maxSynapses) {
   setupSynapses(numNeurons, maxSynapses);
}

AllSTDPSynapses::~AllSTDPSynapses() {
   BGSIZE maxTotalSynapses = maxSynapsesPerNeuron_ * countNeurons_;

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
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 */
void AllSTDPSynapses::setupSynapses() {
   setupSynapses(Simulator::getInstance().getTotalNeurons(), Simulator::getInstance().getMaxSynapsesPerNeuron());
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  numNeurons   Total number of neurons in the network.
 *  @param  maxSynapses  Maximum number of synapses per neuron.
 */
void AllSTDPSynapses::setupSynapses(const int numNeurons, const int maxSynapses) {
   AllSpikingSynapses::setupSynapses(numNeurons, maxSynapses);

   BGSIZE maxTotalSynapses = maxSynapses * numNeurons;

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

/**
 *  Loads out all parameters from the config file.
 *  Registered to OperationManager as Operation::loadParameters
 */
void AllSTDPSynapses::loadParameters() {
   AllSpikingSynapses::loadParameters();
   ParameterManager::getInstance().getBGFloatByXpath("//STDPgap/text()", STDPgap);
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
/**
 *  Prints out all parameters to logging file.
 *  Registered to OperationManager as Operation::printParameters
 */
void AllSTDPSynapses::printParameters() const {
   AllSpikingSynapses::printParameters();

   LOG4CPLUS_DEBUG(fileLogger_, "\n\t---AllSTDPSynapses Parameters---" << endl
                                          << "\tEdges type: AllSTDPSynapses" << endl << endl
                  
               <<"\tSTDP gap"<<STDPgap<<endl
               << "\n\tTauspost value: [" << " I: " << tauspost_I_ << ", " << " E: " << tauspost_E_  << "]"<< endl
               << "\n\tTauspre value: [" << " I: " << tauspre_I_ << ", " << " E: " << tauspre_E_ << "]"<< endl
               << "\n\tTaupos value: [" << " I: " << taupos_I_ << ", " << " E: " << taupos_E_  << "]"<< endl
               << "\n\tTau negvalue: [" << " I: " << tauneg_I_ << ", " << " E: " << tauneg_E_  << "]"<< endl
               << "\n\tWex value: [" << " I: " << Wex_I_ << ", " << " E: " << Wex_E_  << "]"<< endl
               << "\n\tAneg value: [" << " I: " << Aneg_I_ << ", " << " E: " << Aneg_E_  << "]"<< endl
               << "\n\tApos value: [" << " I: " << Apos_I_ << ", " << " E: " << Apos_E_  << "]"<< endl
               );

   //cout<< "\tTauspost values: ["  << " I: " << tauspost_I_ << ", " << " E: " << tauspre_E_ << endl;
}

/*
 *  Create a Synapse and connect it to the model.
 *
 *  @param  synapses    The synapse list to reference.
 *  @param  iSyn        Index of the synapse to set.
 *  @param  srcNeuron   Coordinates of the source Neuron.
 *  @param  destNeuron  Coordinates of the destination Neuron.
 *  @param  sumPoint    Summation point address.
 *  @param  deltaT      Inner simulation step duration.
 *  @param  type        Type of the Synapse to create.
 */
void AllSTDPSynapses::createSynapse(const BGSIZE iSyn, int srcNeuron, int destNeuron, BGFLOAT *sumPoint,
                                    const BGFLOAT deltaT, synapseType type) {

   totalDelayPost_[iSyn] = 0;// Apr 12th 2020 move this line so that when AllSpikingSynapses::createSynapse() is called, inside this method the initSpikeQueue() method can be called successfully
   AllSpikingSynapses::createSynapse(iSyn, srcNeuron, destNeuron, sumPoint, deltaT, type);

   // May 1st 2020
   // Use constants from Froemke and Dan (2002).
   // Spike-timing-dependent synaptic modification induced by natural spike trains. Nature 416 (3/2002)
   //Apos_[iSyn] = 0.005;
   //Aneg_[iSyn] = -(1.05*0.005);
   Apos_[iSyn] = Apos_E_;
   Aneg_[iSyn] = Aneg_E_;
   STDPgap_[iSyn] = STDPgap;
   tauspost_[iSyn] = tauspost_E_;
   tauspre_[iSyn] = tauspre_E_;
   taupos_[iSyn] = taupos_E_;
   tauneg_[iSyn] = tauneg_E_;
   Wex_[iSyn] = Wex_E_ ;// this is based on overlap of 2 neurons' radii (r=4) of outgrowth, scale it by SYNAPSE_STRENGTH_ADJUSTMENT.
   mupos_[iSyn] = 0;
   muneg_[iSyn] = 0;

}

#if !defined(USE_GPU)

/*
 *  Advance one specific Synapse.
 *
 *  @param  iSyn      Index of the Synapse to connect to.
 *  @param  neurons   The Neuron list to search from.
 */
void AllSTDPSynapses::advanceSynapse(const BGSIZE iSyn, IAllNeurons *neurons) {
   // If the synapse is inhibitory or its weight is zero, update synapse state using AllSpikingSynapses::advanceSynapse method
  //LOG4CPLUS_FATAL(fileLogger_, "iSyn : " << iSyn );
  
   BGFLOAT &W = this->W_[iSyn];
   if (W = 0.0) {
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
     
      BGFLOAT &tauspre_ = this->tauspre_[iSyn];
      BGFLOAT &tauspost_ = this->tauspost_[iSyn];
      BGFLOAT &taupos_ = this->taupos_[iSyn];
      BGFLOAT &tauneg_ = this->tauneg_[iSyn];
      int &total_delay = this->totalDelay_[iSyn];

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
   
         epre = 1.0;
         epost=1;
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
                   << "\tiSyn: " << iSyn << endl
                   << "\tidxPre: " << idxPre << endl
                   << "\tidxPost: " << idxPost << endl
                   << "\tspikeHistory: " << spikeHistory << endl
                    << "\tepre: " << epre << endl
                   << "\tepost: " << epost << endl
                   << "\tg_simulationStep: " << g_simulationStep << endl
                   << "\tdelta: " << delta << endl << endl);
            */

            if (delta <= -3.0 * tauneg_)
               break;
          
            stdpLearning(iSyn, delta, epost, epre, idxPre, idxPost);
            --offIndex;
         }
 
         changePSR(iSyn, deltaT);
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
                   << "\tiSyn: " << iSyn << endl
                   << "\tidxPre: " << idxPre << endl
                   << "\tidxPost: " << idxPost << endl
                   << "\tspikeHistory: " << spikeHistory << endl
                   << "\tg_simulationStep: " << g_simulationStep << endl
                    << "\tepre: " << epre << endl
                   << "\tepost: " << epost << endl
                   << "\tdelta: " << delta << endl << endl);
            */
            if (delta >= 3.0 * taupos_)
               break;
           
            stdpLearning(iSyn, delta, epost, epre, idxPre, idxPost);
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

BGFLOAT AllSTDPSynapses::synapticWeightModification(const BGSIZE iSyn, BGFLOAT synapticWeight, double delta)
{
   BGFLOAT STDPgap_ = this->STDPgap_[iSyn];
   BGFLOAT muneg_ = this->muneg_[iSyn];
   BGFLOAT mupos_ = this->mupos_[iSyn];
   BGFLOAT tauneg_ = this->tauneg_[iSyn];
   BGFLOAT taupos_ = this->taupos_[iSyn];
   BGFLOAT Aneg_ = this->Aneg_[iSyn];
   BGFLOAT Apos_ = this->Apos_[iSyn];
   BGFLOAT Wex_ = this->Wex_[iSyn];
   BGFLOAT &W = this->W_[iSyn];
   synapseType type = this->type_[iSyn];
   BGFLOAT dw;
   BGFLOAT oldW=W;

   BGFLOAT modDelta;
    if(delta<0)
      modDelta=delta*-1;
   else
   {
      modDelta=delta;
   }
   

   if (delta < -STDPgap_) {
      // depression
      
      dw = synapticWeightModification(iSyn, W, delta);// normalize
   } else if (delta > STDPgap_) {
      // potentiation
      dw = synapticWeightModification(iSyn, W, delta); // normalize
   } 
   return dw;
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
void AllSTDPSynapses::stdpLearning(const BGSIZE iSyn, double delta, double epost, double epre, int srcN, int destN ) {
    
   BGFLOAT STDPgap_ = this->STDPgap_[iSyn];
   BGFLOAT muneg_ = this->muneg_[iSyn];
   BGFLOAT mupos_ = this->mupos_[iSyn];
   BGFLOAT tauneg_ = this->tauneg_[iSyn];
   BGFLOAT taupos_ = this->taupos_[iSyn];
   BGFLOAT Aneg_ = this->Aneg_[iSyn];
   BGFLOAT Apos_ = this->Apos_[iSyn];
   BGFLOAT Wex_ = this->Wex_[iSyn];
   BGFLOAT &W = this->W_[iSyn];
   synapseType type = this->type_[iSyn];
   BGFLOAT dw;
   BGFLOAT oldW=W;
   BGFLOAT modDelta;
    if(delta<0)
      modDelta=delta*-1;
   else
   {
      modDelta=delta;
   }
   

   if (delta < -STDPgap_) {
      // depression
      
      dw = pow(fabs(W) / Wex_, muneg_) * Aneg_ * exp(delta / tauneg_);  // normalize
   } else if (delta > STDPgap_) {
      // potentiation
      dw = pow(fabs(Wex_ - fabs(W)) / Wex_, mupos_) * Apos_ * exp(-delta / taupos_); // normalize
   } else {  

      return;
   }


   // dw is the fractional change in synaptic strength; add 1.0 to become the scaling ratio
   //dw = 1.0 + dw * epre * epost;
   dw=1+dw;
  

   // if scaling ratio is less than zero, set it to zero so this synapse, its strength is always zero
 

   // current weight multiplies dw (scaling ratio) to generate new weight
   if(dw!=0)
      W *= dw;


   // if new weight is bigger than Wex_ (maximum allowed weight), then set it to Wex_
   if (fabs(W) > Wex_) {
      W = synSign(type) * Wex_;
   }
/*
            LOG4CPLUS_DEBUG(synapseLogger_,
                   iSyn 
                   << ";" << srcN 
                   << ";" << destN
                   << ";" << delta 
                   <<";"<<oldW
                   << ";" << W
                   <<","<<endl);

     */ 
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


/*
 *  Reset time varying state vars and recompute decay.
 *
 *  @param  iSyn            Index of the synapse to set.
 *  @param  deltaT          Inner simulation step duration
 */
void AllSTDPSynapses::resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT) {
   AllSpikingSynapses::resetSynapse(iSyn, deltaT);
}

