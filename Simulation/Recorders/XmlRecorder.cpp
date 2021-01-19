/*
 * @file XmlRecorder.cpp
 *
 * @ingroup Simulation/Recorders
 * 
 * @brief An implementation for recording spikes history on xml file
 */

#include <functional>

#include "XmlRecorder.h"
#include "AllIFNeurons.h"      // TODO: remove LIF model specific code
#include "ConnGrowth.h"
#include "OperationManager.h"
#include "ParameterManager.h"

// The constructor and destructor
XmlRecorder::XmlRecorder() :
      burstinessHist_(MATRIX_TYPE, MATRIX_INIT, 1, static_cast<int>(Simulator::getInstance().getEpochDuration() *
                                                                    Simulator::getInstance().getNumEpochs()), 0),
      spikesHistory_(MATRIX_TYPE, MATRIX_INIT, 1, static_cast<int>(Simulator::getInstance().getEpochDuration() *
                                                                   Simulator::getInstance().getNumEpochs() * 100), 0) {
   resultFileName_ = Simulator::getInstance().getResultFileName();

   function<void()> printParametersFunc = std::bind(&XmlRecorder::printParameters, this);
   OperationManager::getInstance().registerOperation(Operations::printParameters, printParametersFunc);

   fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));
}

XmlRecorder::~XmlRecorder() {
}

/// Initialize data
/// Create a new xml file.
///
/// @param[in] stateOutputFileName	File name to save histories
void XmlRecorder::init() {
   stateOut_.open(resultFileName_.c_str());
}

/// Init radii and rates history matrices with default values
void XmlRecorder::initDefaultValues() {
}

/// Init radii and rates history matrices with current radii and rates
void XmlRecorder::initValues() {
}

/// Get the current radii and rates values
void XmlRecorder::getValues() {
}

/// Terminate process
void XmlRecorder::term() {
   stateOut_.close();
}

/// Compile history information in every epoch
///
/// @param[in] neurons 	The entire list of neurons.
void XmlRecorder::compileHistories(IAllNeurons &neurons) {
   AllSpikingNeurons &spNeurons = dynamic_cast<AllSpikingNeurons &>(neurons);
   int maxSpikes = (int) ((Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getMaxFiringRate()));

   // output spikes
   for (int iNeuron = 0; iNeuron < Simulator::getInstance().getTotalNeurons(); iNeuron++) {
      uint64_t *pSpikes = spNeurons.spikeHistory_[iNeuron];

      int &spikeCount = spNeurons.spikeCount_[iNeuron];
      int &offset = spNeurons.spikeCountOffset_[iNeuron];
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
}

/// Writes simulation results to an output destination.
///
/// @param  neurons the Neuron list to search from.
void XmlRecorder::saveSimData(const IAllNeurons &neurons) {
   // create Neuron Types matrix
   VectorMatrix neuronTypes(MATRIX_TYPE, MATRIX_INIT, 1, Simulator::getInstance().getTotalNeurons(), EXC);
   for (int i = 0; i < Simulator::getInstance().getTotalNeurons(); i++) {
      neuronTypes[i] = Simulator::getInstance().getModel()->getLayout()->neuronTypeMap_[i];
   }

   // create neuron threshold matrix
   VectorMatrix neuronThresh(MATRIX_TYPE, MATRIX_INIT, 1, Simulator::getInstance().getTotalNeurons(), 0);
   for (int i = 0; i < Simulator::getInstance().getTotalNeurons(); i++) {
      neuronThresh[i] = dynamic_cast<const AllIFNeurons &>(neurons).Vthresh_[i];
   }

   // Write XML header information:
   stateOut_ << "<?xml version=\"1.0\" standalone=\"no\"?>\n"
             << "<!-- State output file for the DCT growth modeling-->\n";
   //stateOut << version; TODO: version
   auto layout = Simulator::getInstance().getModel()->getLayout();

   // Write the core state information:
   stateOut_ << "<SimState>\n";
   stateOut_ << "   " << burstinessHist_.toXML("burstinessHist") << endl;
   stateOut_ << "   " << spikesHistory_.toXML("spikesHistory") << endl;
   stateOut_ << "   " << layout->xloc_->toXML("xloc") << endl;
   stateOut_ << "   " << layout->yloc_->toXML("yloc") << endl;
   stateOut_ << "   " << neuronTypes.toXML("neuronTypes") << endl;

   // create starter neurons matrix
   int num_starter_neurons = static_cast<int>(layout->numEndogenouslyActiveNeurons_);
   if (num_starter_neurons > 0) {
      VectorMatrix starterNeurons(MATRIX_TYPE, MATRIX_INIT, 1, num_starter_neurons);
      getStarterNeuronMatrix(starterNeurons, layout->starterMap_);
      stateOut_ << "   " << starterNeurons.toXML("starterNeurons") << endl;
   }

   // Write neuron threshold
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

///  Get starter Neuron matrix.
///
///  @param  matrix      Starter Neuron matrix.
///  @param  starter_map Bool map to reference neuron matrix location from.
void XmlRecorder::getStarterNeuronMatrix(VectorMatrix &matrix, const bool *starterMap) {
   int cur = 0;
   for (int i = 0; i < Simulator::getInstance().getTotalNeurons(); i++) {
      if (starterMap[i]) {
         matrix[cur] = i;
         cur++;
      }
   }
}

///  Prints out all parameters to logging file.
///  Registered to OperationManager as Operation::printParameters
void XmlRecorder::printParameters() {
   LOG4CPLUS_DEBUG(fileLogger_, "\nXMLRECORDER PARAMETERS" << endl
                                      << "\tResult file path: " << resultFileName_ << endl
                                      << "\tBurstiness History Size: " << burstinessHist_.Size() << endl
                                      << "\tSpikes History Size: " << spikesHistory_.Size() << endl);
}
