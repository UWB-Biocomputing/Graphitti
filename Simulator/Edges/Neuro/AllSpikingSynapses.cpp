/**
 * @file AllSpikingSynapses.cpp
 * 
 * @ingroup Simulator/Edges
 *
 * @brief A container of all dynamic STDP synapse data
 */

#include <iomanip>

#include "AllSpikingSynapses.h"
#include "ParameterManager.h"

using namespace std;

AllSpikingSynapses::AllSpikingSynapses() : AllNeuroEdges() {
   decay_ = nullptr;
   totalDelay_ = nullptr;
   delayQueue_ = nullptr;
   delayIndex_ = nullptr;
   delayQueueLength_ = nullptr;
   tau_ = nullptr;
   tau_II_=0;
   tau_IE_=0;
   tau_EI_=0;
   tau_EE_=0;
   delay_II_=0;
   delay_IE_=0;
   delay_EI_=0;
   delay_EE_=0;
}

AllSpikingSynapses::AllSpikingSynapses(const int numVertices, const int maxEdges) {
   setupEdges(numVertices, maxEdges);
}

AllSpikingSynapses::~AllSpikingSynapses() {
   BGSIZE maxTotalSynapses = maxEdgesPerVertex_ * countVertices_;
  
  if (maxTotalSynapses != 0) {
      delete[] decay_;
      delete[] totalDelay_;
      delete[] delayQueue_;
      delete[] delayIndex_;
      delete[] delayQueueLength_;
      delete[] tau_;
  }

   decay_ = nullptr;
   totalDelay_ = nullptr;
   delayQueue_ = nullptr;
   delayIndex_ = nullptr;
   delayQueueLength_ = nullptr;
   tau_ = nullptr;
}

///  Setup the internal structure of the class (allocate memories and initialize them).
void AllSpikingSynapses::setupEdges() {
   setupEdges(Simulator::getInstance().getTotalVertices(), Simulator::getInstance().getMaxEdgesPerVertex());
}

///  Setup the internal structure of the class (allocate memories and initialize them).
///
///  @param  numVertices   Total number of vertices in the network.
///  @param  maxEdges  Maximum number of synapses per neuron.
void AllSpikingSynapses::setupEdges(const int numVertices, const int maxEdges) {
   AllNeuroEdges::setupEdges(numVertices, maxEdges);

   BGSIZE maxTotalSynapses = maxEdges * numVertices;

   if (maxTotalSynapses != 0) {
      decay_ = new BGFLOAT[maxTotalSynapses];
      totalDelay_ = new int[maxTotalSynapses];
      delayQueue_ = new uint32_t[maxTotalSynapses];
      delayIndex_ = new int[maxTotalSynapses];
      delayQueueLength_ = new int[maxTotalSynapses];
      tau_ = new BGFLOAT[maxTotalSynapses];
   }
}

///  Initializes the queues for the Synapse.
///
///  @param  iEdg   index of the synapse to set.
void AllSpikingSynapses::initSpikeQueue(const BGSIZE iEdg) {
   int &totalDelay = totalDelay_[iEdg];
   uint32_t &delayQueue = delayQueue_[iEdg];
   int &delayIdx = delayIndex_[iEdg];
   int &ldelayQueue = delayQueueLength_[iEdg];

   uint32_t size = totalDelay / (sizeof(uint8_t) * 8) + 1;
   assert(size <= BYTES_OF_DELAYQUEUE);
   delayQueue = 0;
   delayIdx = 0;
   ldelayQueue = LENGTH_OF_DELAYQUEUE;
}

///  Reset time varying state vars and recompute decay.
///
///  @param  iEdg     Index of the synapse to set.
///  @param  deltaT   Inner simulation step duration
void AllSpikingSynapses::resetEdge(const BGSIZE iEdg, const BGFLOAT deltaT) {
   AllNeuroEdges::resetEdge(iEdg, deltaT);

   assert(updateDecay(iEdg, deltaT));
}


void AllSpikingSynapses::loadParameters() {
   ParameterManager::getInstance().getBGFloatByXpath("//tau/ii/text()", tau_II_);
   ParameterManager::getInstance().getBGFloatByXpath("//tau/ie/text()", tau_IE_);
   ParameterManager::getInstance().getBGFloatByXpath("//tau/ei/text()", tau_EI_);
   ParameterManager::getInstance().getBGFloatByXpath("//tau/ee/text()", tau_EE_);
   ParameterManager::getInstance().getBGFloatByXpath("//delay/ii/text()", delay_II_);
   ParameterManager::getInstance().getBGFloatByXpath("//delay/ie/text()", delay_IE_);
   ParameterManager::getInstance().getBGFloatByXpath("//delay/ei/text()", delay_EI_);
   ParameterManager::getInstance().getBGFloatByXpath("//delay/ee/text()", delay_EE_);
}

///  Prints out all parameters to logging file.
///  Registered to OperationManager as Operation::printParameters
void AllSpikingSynapses::printParameters() const {
   AllNeuroEdges::printParameters();
   
   LOG4CPLUS_DEBUG(edgeLogger_, "\n\t---AllSpikingSynapses Parameters---" << endl
                   << "\tEdges type: AllSpikingSynapses" << endl << endl);
   LOG4CPLUS_DEBUG(edgeLogger_, "\n\tTau values: ["
                   << " II: " << tau_II_ << ", " << " IE: " << tau_IE_ << "," << "EI : " << tau_EI_<< "," << " EE: " << tau_EE_ << "]"
                   << endl);
   
   LOG4CPLUS_DEBUG(edgeLogger_,"\n\tDelay values: ["
                   << " II: "<< delay_II_ << ", " << " IE: "<< delay_IE_ << "," << "EI :" << delay_EI_<< "," << " EE: "<< delay_EE_ << "]"
                   << endl);
}

///  Sets the data for Synapse to input's data.
///
///  @param  input  istream to read from.
///  @param  iEdg   Index of the synapse to set.
void AllSpikingSynapses::readEdge(istream &input, const BGSIZE iEdg) {
   AllNeuroEdges::readEdge(input, iEdg);

   // input.ignore() so input skips over end-of-line characters.
   input >> decay_[iEdg];
   input.ignore();
   input >> totalDelay_[iEdg];
   input.ignore();
   input >> delayQueue_[iEdg];
   input.ignore();
   input >> delayIndex_[iEdg];
   input.ignore();
   input >> delayQueueLength_[iEdg];
   input.ignore();
   input >> tau_[iEdg];
   input.ignore();
}

///  Write the synapse data to the stream.
///
///  @param  output  stream to print out to.
///  @param  iEdg    Index of the synapse to print out.
void AllSpikingSynapses::writeEdge(ostream &output, const BGSIZE iEdg) const {
   AllNeuroEdges::writeEdge(output, iEdg);

   output << decay_[iEdg] << ends;
   output << totalDelay_[iEdg] << ends;
   output << delayQueue_[iEdg] << ends;
   output << delayIndex_[iEdg] << ends;
   output << delayQueueLength_[iEdg] << ends;
   output << tau_[iEdg] << ends;
}

///  Create a Synapse and connect it to the model.
///
///  @param  iEdg        Index of the synapse to set.
///  @param  srcVertex   Coordinates of the source Neuron.
///  @param  destVertex  Coordinates of the destination Neuron.
///  @param  sumPoint    Summation point address.
///  @param  deltaT      Inner simulation step duration.
///  @param  type        Type of the Synapse to create.
void AllSpikingSynapses::createEdge(const BGSIZE iEdg, int srcVertex, int destVertex, BGFLOAT *sumPoint,
                                       const BGFLOAT deltaT, edgeType type) {
   BGFLOAT delay;

   inUse_[iEdg] = true;
   summationPoint_[iEdg] = sumPoint;
   destVertexIndex_[iEdg] = destVertex;
   sourceVertexIndex_[iEdg] = srcVertex;
   W_[iEdg] = edgSign(type) * 10.0e-9;
   type_[iEdg] = type;
   tau_[iEdg] = DEFAULT_tau;

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

   tau_[iEdg] = tau;
   totalDelay_[iEdg] = static_cast<int>( delay / deltaT ) + 1;

   // initializes the queues for the Synapses
   initSpikeQueue(iEdg);
   // reset time varying state vars and recompute decay
   resetEdge(iEdg, deltaT);
}

#if !defined(USE_GPU)

///  Checks if there is an input spike in the queue.
///
///  @param  iEdg   Index of the Synapse to connect to.
///  @return true if there is an input spike event.
bool AllSpikingSynapses::isSpikeQueue(const BGSIZE iEdg) {
   uint32_t &delayQueue = delayQueue_[iEdg];
   int &delayIdx = delayIndex_[iEdg];
   int &ldelayQueue = delayQueueLength_[iEdg];

   bool r = delayQueue & (0x1 << delayIdx);
   delayQueue &= ~(0x1 << delayIdx);
   if (++delayIdx >= ldelayQueue) {
      delayIdx = 0;
   }
   return r;
}

///  Prepares Synapse for a spike hit.
///
///  @param  iEdg   Index of the Synapse to update.
void AllSpikingSynapses::preSpikeHit(const BGSIZE iEdg) {
   uint32_t &delayQueue = delayQueue_[iEdg];
   int &delayIdx = delayIndex_[iEdg];
   int &ldelayQueue = delayQueueLength_[iEdg];
   int &totalDelay = totalDelay_[iEdg];

   // Add to spike queue

   // calculate index where to insert the spike into delayQueue
   //LOG4CPLUS_TRACE(edgeLogger_,"delayidx "<<delayIdx<<" totalDelay "<<totalDelay<<" ldelayQueue "<<ldelayQueue);
   int idx = delayIdx + totalDelay;
   if (idx >= ldelayQueue) {
      idx -= ldelayQueue;
   }
   if((delayQueue & (0x1 << idx)) != 0)
   {
      LOG4CPLUS_ERROR(edgeLogger_,"Delay Queue Error " << setbase(2) << delayQueue << setbase(10) << " index " << idx << " edge ID " << iEdg);
      assert(false);
   }
   
   // set a spike
   delayQueue |= (0x1 << idx);
}

///  Prepares Synapse for a spike hit (for back propagation).
///
///  @param  iEdg   Index of the Synapse to update.
void AllSpikingSynapses::postSpikeHit(const BGSIZE iEdg) {
}

///  Advance one specific Synapse.
///
///  @param  iEdg      Index of the Synapse to connect to.
///  @param  neurons   The Neuron list to search from.
void AllSpikingSynapses::advanceEdge(const BGSIZE iEdg, AllVertices *neurons) {
   BGFLOAT &decay = decay_[iEdg];
   BGFLOAT &psr = psr_[iEdg];
   BGFLOAT &summationPoint = *(summationPoint_[iEdg]);

   // is an input in the queue?
   if (isSpikeQueue(iEdg)) {
      changePSR(iEdg, Simulator::getInstance().getDeltaT());
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

///  Calculate the post synapse response after a spike.
///
///  @param  iEdg        Index of the synapse to set.
///  @param  deltaT      Inner simulation step duration.
void AllSpikingSynapses::changePSR(const BGSIZE iEdg, const BGFLOAT deltaT) {
   BGFLOAT &psr = psr_[iEdg];
   BGFLOAT &W = W_[iEdg];
   BGFLOAT &decay = decay_[iEdg];

   psr += (W / decay);    // calculate psr
}

#endif //!defined(USE_GPU)

///  Updates the decay if the synapse selected.
///
///  @param  iEdg    Index of the synapse to set.
///  @param  deltaT  Inner simulation step duration
bool AllSpikingSynapses::updateDecay(const BGSIZE iEdg, const BGFLOAT deltaT) {
   BGFLOAT &tau = tau_[iEdg];
   BGFLOAT &decay = decay_[iEdg];

   if (tau > 0) {
      decay = exp(-deltaT / tau);
      return true;
   }
   return false;
}

///  Check if the back propagation (notify a spike event to the pre neuron)
///  is allowed in the synapse class.
///
///  @return true if the back propagation is allowed.
bool AllSpikingSynapses::allowBackPropagation() {
   return false;
}

///  Prints SynapsesProps data to console.
void AllSpikingSynapses::printSynapsesProps() const {
   AllNeuroEdges::printSynapsesProps();
   for (int i = 0; i < maxEdgesPerVertex_ * countVertices_; i++) {
      if (W_[i] != 0.0) {
         cout << "decay[" << i << "] = " << decay_[i];
         cout << " tau: " << tau_[i];
         cout << " total_delay: " << totalDelay_[i] << endl;
      }
   }
}
