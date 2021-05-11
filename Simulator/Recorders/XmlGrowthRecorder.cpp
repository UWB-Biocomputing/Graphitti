/**
 * @file XmlGrowthRecorder.cpp
 *
 * @ingroup Simulator/Recorders
 *
 * @brief An implementation for recording spikes history in an XML file for growth simulations
 */

#include "XmlGrowthRecorder.h"
#include "Simulator.h"
#include "Model.h"
#include "AllIFNeurons.h"      // TODO: remove LIF model specific code
#include "ConnGrowth.h"

//! The constructor and destructor
XmlGrowthRecorder::XmlGrowthRecorder() :
      XmlRecorder(),
      burstinessHist_(MATRIX_TYPE, MATRIX_INIT, 1, static_cast<int>(Simulator::getInstance().getEpochDuration() *
                                                                    Simulator::getInstance().getNumEpochs()), 0),
      spikesHistory_(MATRIX_TYPE, MATRIX_INIT, 1, static_cast<int>(Simulator::getInstance().getEpochDuration() *
                                                                   Simulator::getInstance().getNumEpochs() * 100), 0),
      ratesHistory_(MATRIX_TYPE, MATRIX_INIT, static_cast<int>(Simulator::getInstance().getNumEpochs() + 1),
                    Simulator::getInstance().getTotalVertices()),
      radiiHistory_(MATRIX_TYPE, MATRIX_INIT, static_cast<int>(Simulator::getInstance().getNumEpochs() + 1),
                    Simulator::getInstance().getTotalVertices()) {
}

XmlGrowthRecorder::~XmlGrowthRecorder() {
}

/// Init radii and rates history matrices with default values
void XmlGrowthRecorder::initDefaultValues() {
   shared_ptr<Connections> conns = Simulator::getInstance().getModel()->getConnections();
   BGFLOAT startRadius = dynamic_cast<ConnGrowth *>(conns.get())->growthParams_.startRadius;

   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      radiiHistory_(0, i) = startRadius;
      ratesHistory_(0, i) = 0;
   }
}

/// Init radii and rates history matrices with current radii and rates
void XmlGrowthRecorder::initValues() {
   shared_ptr<Connections> conns = Simulator::getInstance().getModel()->getConnections();

   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      radiiHistory_(0, i) = (*dynamic_cast<ConnGrowth *>(conns.get())->radii_)[i];
      ratesHistory_(0, i) = (*dynamic_cast<ConnGrowth *>(conns.get())->rates_)[i];
   }
}

/// Get the current radii and rates values
void XmlGrowthRecorder::getValues() {
   Connections *conns = Simulator::getInstance().getModel()->getConnections().get();

   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      (*dynamic_cast<ConnGrowth *>(conns)->radii_)[i] = radiiHistory_(Simulator::getInstance().getCurrentStep(), i);
      (*dynamic_cast<ConnGrowth *>(conns)->rates_)[i] = ratesHistory_(Simulator::getInstance().getCurrentStep(), i);
   }
}

/// Compile growth information in every epoch
///
/// @param[in] neurons 	The entire list of neurons.
void XmlGrowthRecorder::compileGrowthHistories(IAllVertices &neurons) {
   shared_ptr<Connections> conns = Simulator::getInstance().getModel()->getConnections();

   BGFLOAT minRadius = dynamic_cast<ConnGrowth *>(conns.get())->growthParams_.minRadius;
   VectorMatrix &rates = (*dynamic_cast<ConnGrowth *>(conns.get())->rates_);
   VectorMatrix &radii = (*dynamic_cast<ConnGrowth *>(conns.get())->radii_);

   for (int iVertex = 0; iVertex < Simulator::getInstance().getTotalVertices(); iVertex++) {
      // record firing rate to history matrix
      ratesHistory_(Simulator::getInstance().getCurrentStep(), iVertex) = rates[iVertex];

      // Cap minimum radius size and record radii to history matrix
      // TODO: find out why we cap this here.
      if (radii[iVertex] < minRadius)
         radii[iVertex] = minRadius;

      // record radius to history matrix
      radiiHistory_(Simulator::getInstance().getCurrentStep(), iVertex) = radii[iVertex];
   }
}

/// Compile history information in every epoch
///
/// @param[in] neurons 	The entire list of neurons.
void XmlGrowthRecorder::compileHistories(IAllVertices &neurons) {
   AllSpikingNeurons &spNeurons = dynamic_cast<AllSpikingNeurons &>(neurons);
   int maxSpikes = (int) ((Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getMaxFiringRate()));

   // output spikes
   for (int iVertex = 0; iVertex < Simulator::getInstance().getTotalVertices(); iVertex++) {
      uint64_t *pSpikes = spNeurons.spikeHistory_[iVertex];

      int &spikeCount = spNeurons.spikeCount_[iVertex];
      int &offset = spNeurons.spikeCountOffset_[iVertex];
      for (int i = 0, idxSp = offset; i < spikeCount; i++, idxSp++) {
         // Single precision (float) gives you 23 bits of significand, 8 bits of exponent,
         // and 1 sign bit. Double precision (double) gives you 52 bits of significand,
         // 11 bits of exponent, and 1 sign bit.
         // Therefore, single precision can only handle 2^23 = 8,388,608 simulation steps
         // or 8 epochs (1 epoch = 100s, 1 simulation step = 0.1ms).

         if (idxSp >= maxSpikes) idxSp = 0;
         // compile network wide burstiness index data in 1s bins
         int idx1 = static_cast<int>( static_cast<double>( pSpikes[idxSp] ) * Simulator::getInstance().getDeltaT());
         burstinessHist_[idx1] = burstinessHist_[idx1] + 1.0;

         // compile network wide spike count in 10ms bins
         int idx2 = static_cast<int>( static_cast<double>( pSpikes[idxSp] ) * Simulator::getInstance().getDeltaT() *
                                      100);
         spikesHistory_[idx2] = spikesHistory_[idx2] + 1.0;
      }
   }

   // clear spike count
   spNeurons.clearSpikeCounts();

   compileGrowthHistories(neurons);
}

/// Writes simulation results to an output destination.
///
/// @param  neurons the Neuron list to search from.
void XmlGrowthRecorder::saveSimData(const IAllVertices &neurons) {
   // create Neuron Types matrix
   VectorMatrix neuronTypes(MATRIX_TYPE, MATRIX_INIT, 1, Simulator::getInstance().getTotalVertices(), EXC);
   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      neuronTypes[i] = Simulator::getInstance().getModel()->getLayout()->vertexTypeMap_[i];
   }

   // create neuron threshold matrix
   VectorMatrix neuronThresh(MATRIX_TYPE, MATRIX_INIT, 1, Simulator::getInstance().getTotalVertices(), 0);
   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      neuronThresh[i] = dynamic_cast<const AllIFNeurons &>(neurons).Vthresh_[i];
   }

   // Write XML header information:
   stateOut_ << "<?xml version=\"1.0\" standalone=\"no\"?>\n"
             << "<!-- State output file for the DCT growth modeling-->\n";
   //stateOut << version; TODO: version

   // Write the core state information:
   stateOut_ << "<SimState>\n";
   stateOut_ << "   " << radiiHistory_.toXML("radiiHistory") << endl;
   stateOut_ << "   " << ratesHistory_.toXML("ratesHistory") << endl;
   stateOut_ << "   " << burstinessHist_.toXML("burstinessHist") << endl;
   stateOut_ << "   " << spikesHistory_.toXML("spikesHistory") << endl;
   stateOut_ << "   " << Simulator::getInstance().getModel()->getLayout()->xloc_->toXML("xloc") << endl;
   stateOut_ << "   " << Simulator::getInstance().getModel()->getLayout()->yloc_->toXML("yloc") << endl;
   stateOut_ << "   " << neuronTypes.toXML("neuronTypes") << endl;

   // create starter neuron matrix
   int num_starter_neurons = static_cast<int>(Simulator::getInstance().getModel()->getLayout()->numEndogenouslyActiveNeurons_);
   if (num_starter_neurons > 0) {
      VectorMatrix starterNeurons(MATRIX_TYPE, MATRIX_INIT, 1, num_starter_neurons);
      getStarterNeuronMatrix(starterNeurons, Simulator::getInstance().getModel()->getLayout()->starterMap_);
      stateOut_ << "   " << starterNeurons.toXML("starterNeurons") << endl;
   }

   // Write neuron thresold
   stateOut_ << "   " << neuronThresh.toXML("neuronThresh") << endl;

   // write time between growth cycles
   stateOut_ << "   <Matrix name=\"Tsim\" type=\"complete\" rows=\"1\" columns=\"1\" multiplier=\"1.0\">" << endl;
   stateOut_ << "   " << Simulator::getInstance().getEpochDuration() << endl;
   stateOut_ << "</Matrix>" << endl;

   // write simulation end time
   stateOut_ << "   <Matrix name=\"simulationEndTime\" type=\"complete\" rows=\"1\" columns=\"1\" multiplier=\"1.0\">"
             << endl;
   stateOut_ << "   " << g_simulationStep * Simulator::getInstance().getDeltaT() << endl;
   stateOut_ << "</Matrix>" << endl;
   stateOut_ << "</SimState>" << endl;
}

///  Prints out all parameters to logging file.
///  Registered to OperationManager as Operation::printParameters
void XmlGrowthRecorder::printParameters() {
   LOG4CPLUS_DEBUG(fileLogger_, "\nXMLRECORDER PARAMETERS" << endl
                                      << "\tResult file path: " << resultFileName_ << endl
                                      << "\tBurstiness History Size: " << burstinessHist_.Size() << endl
                                      << "\tSpikes History Size: " << spikesHistory_.Size() << endl
                                      << "\n---XmlGrowthRecorder Parameters---" << endl
                                      << "\tRecorder type: XmlGrowthRecorder" << endl);
}

///  Get starter Neuron matrix.
///
///  @param  matrix      Starter Neuron matrix.
///  @param  starterMap Bool map to reference neuron matrix location from.
void XmlGrowthRecorder::getStarterNeuronMatrix(VectorMatrix &matrix, const bool *starterMap) {
   int cur = 0;
   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      if (starterMap[i]) {
         matrix[cur] = i;
         cur++;
      }
   }
}
