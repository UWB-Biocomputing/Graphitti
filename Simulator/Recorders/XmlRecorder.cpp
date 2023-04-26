/**
 * @file XmlRecorder.cpp
 *
 * @ingroup Simulator/Recorders
 *
 * @brief An implementation for recording spikes history on xml file
 */

#include "XmlRecorder.h"
#include "AllIFNeurons.h"   // TODO: remove LIF model specific code
#include "AllSpikingNeurons.h"
#include "ConnGrowth.h"
#include "OperationManager.h"
#include "ParameterManager.h"
#include "VectorMatrix.h"
#include <functional>

// constructor
// TODO: I believe the initializer for spikesHistory_ assumes a particular deltaT
XmlRecorder::XmlRecorder() :
   spikesHistory_(MATRIX_TYPE, MATRIX_INIT, 1,
                  static_cast<int>(Simulator::getInstance().getEpochDuration()
                                   * Simulator::getInstance().getNumEpochs() * 100),
                  static_cast<BGFLOAT>(0.0))
{
   ParameterManager::getInstance().getStringByXpath(
      "//RecorderParams/RecorderFiles/resultFileName/text()", resultFileName_);
   function<void()> printParametersFunc = std::bind(&XmlRecorder::printParameters, this);
   OperationManager::getInstance().registerOperation(Operations::printParameters,
                                                     printParametersFunc);
   fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));
}

// Create a new xml file and initialize data
/// @param[in] stateOutputFileName      File name to save histories
void XmlRecorder::init()
{
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
void XmlRecorder::initDefaultValues()
{
}

/// Init radii and rates history matrices with current radii and rates
void XmlRecorder::initValues()
{
}

/// Get the current radii and rates values
void XmlRecorder::getValues()
{
}

/// Terminate process
void XmlRecorder::term()
{
   resultOut_.close();
}

/// Compile history information in every epoch
/// @param[in] neurons    The entire list of neurons.
void XmlRecorder::compileHistories(AllVertices &vertices)
{
   AllSpikingNeurons &spNeurons = dynamic_cast<AllSpikingNeurons &>(vertices);
   Simulator &simulator = Simulator::getInstance();
   int maxSpikes = static_cast<int>(simulator.getEpochDuration() * simulator.getMaxFiringRate());

   for (int iNeuron = 0; iNeuron < spNeurons.vertexEvents_.size(); iNeuron++) {
      for (int eventIterator = 0;
           eventIterator < spNeurons.vertexEvents_[iNeuron].getNumEventsInEpoch();
           eventIterator++) {
         // compile network wide spike count in 10ms bins
         int idx2
            = static_cast<int>(static_cast<double>(spNeurons.vertexEvents_[iNeuron][eventIterator])
                               * Simulator::getInstance().getDeltaT() * 100);
         spikesHistory_[idx2] = spikesHistory_[idx2] + 1.0;
      }
   }
   spNeurons.clearSpikeCounts();
}

/// Writes simulation results to an output destination.
/// @param  neurons the Neuron list to search from.
void XmlRecorder::saveSimData(const AllVertices &vertices)
{
   Simulator &simulator = Simulator::getInstance();
   // create Neuron Types matrix
   VectorMatrix neuronTypes(MATRIX_TYPE, MATRIX_INIT, 1, simulator.getTotalVertices(), EXC);
   for (int i = 0; i < simulator.getTotalVertices(); i++) {
      neuronTypes[i] = simulator.getModel().getLayout().vertexTypeMap_[i];
   }
   // create neuron threshold matrix
   VectorMatrix neuronThresh(MATRIX_TYPE, MATRIX_INIT, 1, simulator.getTotalVertices(), 0);
   for (int i = 0; i < simulator.getTotalVertices(); i++) {
      neuronThresh[i] = dynamic_cast<const AllIFNeurons &>(vertices).Vthresh_[i];
   }

   // Write XML header information:
   resultOut_ << "<?xml version=\"1.0\" standalone=\"no\"?>\n"
              << "<!-- State output file for the DCT growth modeling-->\n";
   // stateOut << version; TODO: version
   auto &layout = simulator.getModel().getLayout();

   // Write the core state information:
   resultOut_ << "<SimState>\n";
   resultOut_ << "   " << spikesHistory_.toXML("spikesHistory") << endl;
   resultOut_ << "   " << layout.xloc_.toXML("xloc") << endl;
   resultOut_ << "   " << layout.yloc_.toXML("yloc") << endl;
   resultOut_ << "   " << neuronTypes.toXML("neuronTypes") << endl;

   // create starter neurons matrix
   int num_starter_neurons = static_cast<int>(layout.numEndogenouslyActiveNeurons_);
   if (num_starter_neurons > 0) {
      VectorMatrix starterNeurons(MATRIX_TYPE, MATRIX_INIT, 1, num_starter_neurons);
      getStarterNeuronMatrix(starterNeurons, layout.starterMap_);
      resultOut_ << "   " << starterNeurons.toXML("starterNeurons") << endl;
   }

   // Write neuron threshold
   resultOut_ << "   " << neuronThresh.toXML("neuronThresh") << endl;

   // write epoch duration
   resultOut_
      << "   <Matrix name=\"Tsim\" type=\"complete\" rows=\"1\" columns=\"1\" multiplier=\"1.0\">"
      << endl;
   resultOut_ << "   " << simulator.getEpochDuration() << endl;
   resultOut_ << "</Matrix>" << endl;

   // write simulation end time
   resultOut_
      << "   <Matrix name=\"simulationEndTime\" type=\"complete\" rows=\"1\" columns=\"1\" multiplier=\"1.0\">"
      << endl;
   resultOut_ << "   " << g_simulationStep * simulator.getDeltaT() << endl;
   resultOut_ << "</Matrix>" << endl;
   resultOut_ << "</SimState>" << endl;
}

/*
 *  Get starter Neuron matrix.
 *  @param  matrix      Starter Neuron matrix.
 *  @param  starter_map Bool map to reference neuron matrix location from.
 */
void XmlRecorder::getStarterNeuronMatrix(VectorMatrix &matrix, const std::vector<bool> &starterMap)
{
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
void XmlRecorder::printParameters()
{
   LOG4CPLUS_DEBUG(fileLogger_, "\nXMLRECORDER PARAMETERS"
                                   << endl
                                   << "\tResult file path: " << resultFileName_ << endl
                                   << "\tSpikes History Size: " << spikesHistory_.Size() << endl);
}
