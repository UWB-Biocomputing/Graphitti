#include "AllDSSynapses.h"
#include "ParameterManager.h"
#include "OperationManager.h"

AllDSSynapses::AllDSSynapses() : AllSpikingSynapses() {
   lastSpike_ = NULL;
   r_ = NULL;
   u_ = NULL;
   D_ = NULL;
   U_ = NULL;
   F_ = NULL;
   U_II_= 0;
   U_IE_= 0;
   U_EI_= 0;
   U_EE_= 0;

   D_II_= 0;
   D_IE_= 0;
   D_EI_= 0;
   D_EE_= 0;

   F_II_= 0;
   F_IE_= 0;
   F_EI_= 0;
   F_EE_= 0;
}

AllDSSynapses::AllDSSynapses(const int num_neurons, const int max_synapses) :
      AllSpikingSynapses(num_neurons, max_synapses) {
   setupSynapses(num_neurons, max_synapses);
}

AllDSSynapses::~AllDSSynapses() {
   cleanupSynapses();
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 */
void AllDSSynapses::setupSynapses() {
   setupSynapses(Simulator::getInstance().getTotalNeurons(), Simulator::getInstance().getMaxSynapsesPerNeuron());
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  num_neurons   Total number of neurons in the network.
 *  @param  max_synapses  Maximum number of synapses per neuron.
 */
void AllDSSynapses::setupSynapses(const int num_neurons, const int max_synapses) {
   AllSpikingSynapses::setupSynapses(num_neurons, max_synapses);

   BGSIZE max_total_synapses = max_synapses * num_neurons;

   if (max_total_synapses != 0) {
      lastSpike_ = new uint64_t[max_total_synapses];
      r_ = new BGFLOAT[max_total_synapses];
      u_ = new BGFLOAT[max_total_synapses];
      D_ = new BGFLOAT[max_total_synapses];
      U_ = new BGFLOAT[max_total_synapses];
      F_ = new BGFLOAT[max_total_synapses];
      U_II_= 0;
      U_IE_= 0;
      U_EI_= 0;
      U_EE_= 0;

      D_II_= 0;
      D_IE_= 0;
      D_EI_= 0;
      D_EE_= 0;

      F_II_= 0;
      F_IE_= 0;
      F_EI_= 0;
      F_EE_= 0;
   }
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllDSSynapses::cleanupSynapses() {
   BGSIZE max_total_synapses = maxSynapsesPerNeuron_ * countNeurons_;

   if (max_total_synapses != 0) {
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

   AllSpikingSynapses::cleanupSynapses();
}

void AllDSSynapses::loadParameters() {
   ParameterManager::getInstance().getBGFloatByXpath("//U/ii/text()", U_II_);
   ParameterManager::getInstance().getBGFloatByXpath("//U/ie/text()", U_IE_);
   ParameterManager::getInstance().getBGFloatByXpath("//U/ei/text()", U_EI_);
   ParameterManager::getInstance().getBGFloatByXpath("//U/ee/text()", U_EE_);

   ParameterManager::getInstance().getBGFloatByXpath("//D/ii/text()", D_II_);
   ParameterManager::getInstance().getBGFloatByXpath("//D/ie/text()", D_IE_);
   ParameterManager::getInstance().getBGFloatByXpath("//D/ei/text()", D_EI_);
   ParameterManager::getInstance().getBGFloatByXpath("//D/ee/text()", D_EE_);

   ParameterManager::getInstance().getBGFloatByXpath("//F/ii/text()", F_II_);
   ParameterManager::getInstance().getBGFloatByXpath("//F/ie/text()", F_IE_);
   ParameterManager::getInstance().getBGFloatByXpath("//F/ei/text()", F_EI_);
   ParameterManager::getInstance().getBGFloatByXpath("//F/ee/text()", F_EE_);
  
}
/*
 *  Prints out all parameters of the neurons to console.
 */
void AllDSSynapses::printParameters() const {

   AllSpikingSynapses::printParameters();
   cout << "\t*AllDSSynapses Parameters*" << endl;
   cout << "\tEdges type: AllDSSynapses" << endl << endl;
   cout << "\tU values: ["
          << " II: " << U_II_ << ", " << " IE: " << U_IE_ << "," << "EI : " << U_EI_<< "," << " EE: " << U_EE_ << "]"
          << endl;


    cout << "\tD values: ["
          << " II: "<< D_II_ << ", " << " IE: "<< D_IE_ << "," << "EI :" << D_EI_<< "," << " EE: "<< D_EE_ << "]"
          << endl;

   cout << "\tF values: ["
          << " II: "<< F_II_ << ", " << " IE: "<< F_IE_ << "," << "EI :" << F_EI_<< "," << " EE: "<< F_EE_ << "]"
          << endl;
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
 *  @param  synapses    The synapse list to reference.
 *  @param  iSyn        Index of the synapse to set.
 *  @param  source      Coordinates of the source Neuron.
 *  @param  dest        Coordinates of the destination Neuron.
 *  @param  sum_point   Summation point address.
 *  @param  deltaT      Inner simulation step duration.
 *  @param  type        Type of the Synapse to create.
 */
void AllDSSynapses::createSynapse(const BGSIZE iSyn, int source_index, int dest_index, BGFLOAT *sum_point,
                                  const BGFLOAT deltaT, synapseType type) {
   AllSpikingSynapses::createSynapse(iSyn, source_index, dest_index, sum_point, deltaT, type);

   U_[iSyn] = DEFAULT_U;

   BGFLOAT U;
   BGFLOAT D;
   BGFLOAT F;
   switch (type) {
      case II:
         U = U_II_;
         D = D_II_;
         F = F_II_;
         break;
      case IE:
         U = U_IE_;
         D = D_IE_;
         F = F_IE_;
         break;
      case EI:
         U = U_EI_;
         D = D_EI_;
         F = F_EI_;
         break;
      case EE:
         U = U_EE_;
         D = D_EE_;
         F = F_EE_;
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

