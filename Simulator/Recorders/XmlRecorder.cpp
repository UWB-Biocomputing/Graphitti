/**
 * @file XmlRecorder.cpp
 *
 * @ingroup Simulator/Recorders
 * 
 * @brief An implementation for recording spikes history on xml file
 */

#include <functional>

#include "XmlRecorder.h"
#include "AllIFNeurons.h"      // TODO: remove LIF model specific code
#include "ConnGrowth.h"
#include "OperationManager.h"
#include "ParameterManager.h"
#include "VectorMatrix.h"

/// constructor
// TODO: I believe the initializer for spikesHistory_ assumes a particular deltaT
XmlRecorder::XmlRecorder() :
   burstinessHist_(MATRIX_TYPE, MATRIX_INIT, 1, static_cast<int>(Simulator::getInstance().getEpochDuration() *
    Simulator::getInstance().getNumEpochs()), static_cast<BGFLOAT>(0.0)),
   spikesHistory_(MATRIX_TYPE, MATRIX_INIT, 1, static_cast<int>(Simulator::getInstance().getEpochDuration() *
    Simulator::getInstance().getNumEpochs() * 100), static_cast<BGFLOAT>(0.0)) {

   ParameterManager::getInstance().getStringByXpath("//RecorderParams/RecorderFiles/resultFileName/text()", resultFileName_);

   std::function<void()> printParametersFunc = std::bind(&XmlRecorder::printParameters, this);
   OperationManager::getInstance().registerOperation(Operations::printParameters, printParametersFunc);

   fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));
}

   // TODO: Is this needed?
/// destructor
XmlRecorder::~XmlRecorder() {
}

/// Initialize data
/// Create a new xml file.
///
/// @param[in] stateOutputFileName	File name to save histories
void XmlRecorder::init() {
   resultOut_.open(resultFileName_.c_str());

   // TODO: Log error using LOG4CPLUS for workbench
   //       For the time being, we are terminating the program when we can't open a file for writing.
   if (!resultOut_.is_open()) {
      perror("Error opening output file for writing ");
      exit(EXIT_FAILURE);
   }

}

// TODO: for the empty functions below, what should happen? Should they ever
// TODO: be called? Is it an error if they're called?
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
   resultOut_.close();
}

/// Compile history information in every epoch
///
/// @param[in] neurons    The entire list of neurons.
void XmlRecorder::compileHistories(AllVertices &vertices) {
   AllSpikingNeurons &spNeurons = dynamic_cast<AllSpikingNeurons &>(vertices);
   Simulator& simulator = Simulator::getInstance();
   int maxSpikes = static_cast<int>(simulator.getEpochDuration() * simulator.getMaxFiringRate());
   
   // output spikes
   for (int iNeuron = 0; iNeuron < simulator.getTotalVertices(); iNeuron++) {
      uint64_t *pSpikes = spNeurons.spikeHistory_[iNeuron];
      
      const int spikeCount = spNeurons.spikeCount_[iNeuron];
      const int offset = spNeurons.spikeCountOffset_[iNeuron];
      for (int i = 0, idxSp = offset; i < spikeCount; i++, idxSp++) {
         // Single precision (float) gives you 23 bits of significand, 8 bits of exponent,
         // and 1 sign bit. Double precision (double) gives you 52 bits of significand,
         // 11 bits of exponent, and 1 sign bit.
         // Therefore, single precision can only handle 2^23 = 8,388,608 simulation steps
         // or 8 epochs (1 epoch = 100s, 1 simulation step = 0.1ms).
         
         if (idxSp >= maxSpikes) idxSp = 0;
         // compile network wide burstiness index data in 1s bins
         int idx1 = static_cast<int>(static_cast<double>(pSpikes[idxSp]) * simulator.getDeltaT());
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
void XmlRecorder::saveSimData(const AllVertices &vertices) {
   Simulator& simulator = Simulator::getInstance();
   // create Neuron Types matrix
   VectorMatrix neuronTypes(MATRIX_TYPE, MATRIX_INIT, 1, simulator.getTotalVertices(), EXC);
   for (int i = 0; i < simulator.getTotalVertices(); i++) {
      neuronTypes[i] = simulator.getModel()->getLayout()->vertexTypeMap_[i];
   }
   
   // create neuron threshold matrix
   VectorMatrix neuronThresh(MATRIX_TYPE, MATRIX_INIT, 1, simulator.getTotalVertices(), 0);
   for (int i = 0; i < simulator.getTotalVertices(); i++) {
      neuronThresh[i] = dynamic_cast<const AllIFNeurons &>(vertices).Vthresh_[i];
   }
   
   // Write XML header information:
   resultOut_ << "<?xml version=\"1.0\" standalone=\"no\"?>\n"
   << "<!-- State output file for the DCT growth modeling-->\n";
   //stateOut << version; TODO: version
   auto layout = simulator.getModel()->getLayout();
   
   // Write the core state information:
   resultOut_ << "<SimState>\n";
   resultOut_ << "   " << burstinessHist_.toXML("burstinessHist") << std::endl;
   resultOut_ << "   " << spikesHistory_.toXML("spikesHistory") << std::endl;
   resultOut_ << "   " << layout->xloc_->toXML("xloc") << std::endl;
   resultOut_ << "   " << layout->yloc_->toXML("yloc") << std::endl;
   resultOut_ << "   " << neuronTypes.toXML("neuronTypes") << std::endl;
   
   // create starter neurons matrix
   int num_starter_neurons = static_cast<int>(layout->numEndogenouslyActiveNeurons_);
   if (num_starter_neurons > 0) {
      VectorMatrix starterNeurons(MATRIX_TYPE, MATRIX_INIT, 1, num_starter_neurons);
      getStarterNeuronMatrix(starterNeurons, layout->starterMap_);
      resultOut_ << "   " << starterNeurons.toXML("starterNeurons") << std::endl;
   }
   
   // Write neuron threshold
   resultOut_ << "   " << neuronThresh.toXML("neuronThresh") << std::endl;
   
   // write epoch duration
   resultOut_ << "   <Matrix name=\"Tsim\" type=\"complete\" rows=\"1\" columns=\"1\" multiplier=\"1.0\">" << std::endl;
   resultOut_ << "   " << simulator.getEpochDuration() << std::endl;
   resultOut_ << "</Matrix>" << std::endl;
   
   // write simulation end time
   resultOut_ << "   <Matrix name=\"simulationEndTime\" type=\"complete\" rows=\"1\" columns=\"1\" multiplier=\"1.0\">"
   << std::endl;
   resultOut_ << "   " << g_simulationStep * simulator.getDeltaT() << std::endl;
   resultOut_ << "</Matrix>" << std::endl;
   resultOut_ << "</SimState>" << std::endl;
}

/*
 *  Get starter Neuron matrix.
 *
 *  @param  matrix      Starter Neuron matrix.
 *  @param  starter_map Bool map to reference neuron matrix location from.
 */
void XmlRecorder::getStarterNeuronMatrix(VectorMatrix &matrix, const bool *starterMap) {
   int cur = 0;
   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      if (starterMap[i]) {
         matrix[cur] = i;
         cur++;
      }
   }
}

/**
 *  Prints out all parameters to logging file.
 *  Registered to OperationManager as Operation::printParameters
 */
void XmlRecorder::printParameters() {
   LOG4CPLUS_DEBUG(fileLogger_, "\nXMLRECORDER PARAMETERS" << std::endl
                   << "\tResult file path: " << resultFileName_ << std::endl
                   << "\tBurstiness History Size: " << burstinessHist_.Size() << std::endl
                   << "\tSpikes History Size: " << spikesHistory_.Size() << std::endl);
}
