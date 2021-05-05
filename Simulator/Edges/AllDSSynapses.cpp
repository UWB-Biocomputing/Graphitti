#include "AllDSSynapses.h"

AllDSSynapses::AllDSSynapses() : AllSpikingSynapses() {
   lastSpike_ = NULL;
   r_ = NULL;
   u_ = NULL;
   D_ = NULL;
   U_ = NULL;
   F_ = NULL;
}

AllDSSynapses::AllDSSynapses(const int numNeurons, const int maxSynapses) :
      AllSpikingSynapses(numNeurons, maxSynapses) {
   setupSynapses(numNeurons, maxSynapses);
}

AllDSSynapses::~AllDSSynapses() {
   BGSIZE maxTotalSynapses = maxSynapsesPerNeuron_ * countNeurons_;

   if (maxTotalSynapses != 0) {
      delete[] lastSpike_;
      delete[] r_;
      delete[] u_;
      delete[] D_;
      delete[] U_;
      delete[] F_;
   }

   lastSpike_ = NULL;
   r_ = NULL;
   u_ = NULL;
   D_ = NULL;
   U_ = NULL;
   F_ = NULL;
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 */
void AllDSSynapses::setupSynapses() {
   setupSynapses(Simulator::getInstance().getTotalNeurons(), Simulator::getInstance().getMaxSynapsesPerNeuron());
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  numNeurons   Total number of neurons in the network.
 *  @param  maxSynapses  Maximum number of synapses per neuron.
 */
void AllDSSynapses::setupSynapses(const int numNeurons, const int maxSynapses) {
   AllSpikingSynapses::setupSynapses(numNeurons, maxSynapses);

   BGSIZE maxTotalSynapses = maxSynapses * numNeurons;

   if (maxTotalSynapses != 0) {
      lastSpike_ = new uint64_t[maxTotalSynapses];
      r_ = new BGFLOAT[maxTotalSynapses];
      u_ = new BGFLOAT[maxTotalSynapses];
      D_ = new BGFLOAT[maxTotalSynapses];
      U_ = new BGFLOAT[maxTotalSynapses];
      F_ = new BGFLOAT[maxTotalSynapses];
   }
}

/**
 *  Prints out all parameters to logging file.
 *  Registered to OperationManager as Operation::printParameters
 */
void AllDSSynapses::printParameters() const {
   AllSpikingSynapses::printParameters();

   LOG4CPLUS_DEBUG(fileLogger_, "\n\t---AllDSSynapses Parameters---" << endl
                                          << "\tEdges type: AllDSSynapses" << endl << endl);
}


/*
 *  Reset time varying state vars and recompute decay.
 *
 *  @param  iSyn            Index of the synapse to set.
 *  @param  deltaT          Inner simulation step duration
 */
void AllDSSynapses::resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT) {
   AllSpikingSynapses::resetSynapse(iSyn, deltaT);

   u_[iSyn] = DEFAULT_U;
   r_[iSyn] = 1.0;
   lastSpike_[iSyn] = ULONG_MAX;
}

/*
 *  Create a Synapse and connect it to the model.
 *
 *  @param  iSyn        Index of the synapse to set.
 *  @param  srcNeuron      Coordinates of the source Neuron.
 *  @param  destNeuron        Coordinates of the destination Neuron.
 *  @param  sumPoint   Summation point address.
 *  @param  deltaT      Inner simulation step duration.
 *  @param  type        Type of the Synapse to create.
 */
void AllDSSynapses::createSynapse(const BGSIZE iSyn, int srcNeuron, int destNeuron, BGFLOAT *sumPoint,
                                  const BGFLOAT deltaT, synapseType type) {
   AllSpikingSynapses::createSynapse(iSyn, srcNeuron, destNeuron, sumPoint, deltaT, type);

   U_[iSyn] = DEFAULT_U;

   BGFLOAT U;
   BGFLOAT D;
   BGFLOAT F;
   switch (type) {
      case II:
         U = 0.32;
         D = 0.144;
         F = 0.06;
         break;
      case IE:
         U = 0.25;
         D = 0.7;
         F = 0.02;
         break;
      case EI:
         U = 0.05;
         D = 0.125;
         F = 1.2;
         break;
      case EE:
         U = 0.5;
         D = 1.1;
         F = 0.05;
         break;
      default:
         assert(false);
         break;
   }

   this->U_[iSyn] = U;
   this->D_[iSyn] = D;
   this->F_[iSyn] = F;
}

#if !defined(USE_GPU)

/*
 *  Calculate the post synapse response after a spike.
 *
 *  @param  iSyn        Index of the synapse to set.
 *  @param  deltaT      Inner simulation step duration.
 */
void AllDSSynapses::changePSR(const BGSIZE iSyn, const BGFLOAT deltaT) {
   BGFLOAT &psr = this->psr_[iSyn];
   BGFLOAT &W = this->W_[iSyn];
   BGFLOAT &decay = this->decay_[iSyn];
   uint64_t &lastSpike = this->lastSpike_[iSyn];
   BGFLOAT &r = this->r_[iSyn];
   BGFLOAT &u = this->u_[iSyn];
   BGFLOAT &D = this->D_[iSyn];
   BGFLOAT &F = this->F_[iSyn];
   BGFLOAT &U = this->U_[iSyn];

   // adjust synapse parameters
   if (lastSpike != ULONG_MAX) {
      BGFLOAT isi = (g_simulationStep - lastSpike) * deltaT;
      r = 1 + (r * (1 - u) - 1) * exp(-isi / D);
      u = U + u * (1 - U) * exp(-isi / F);
   }
   psr += ((W / decay) * u * r);    // calculate psr
   lastSpike = g_simulationStep;        // record the time of the spike
}

#endif // !defined(USE_GPU)
