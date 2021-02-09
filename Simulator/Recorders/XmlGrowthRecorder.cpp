/**
 * @file XmlGrowthRecorder.cpp
 * 
 * @ingroup Simulation/Recorders
 *
 * @brief An implementation for recording spikes history on xml file
 */

#include "XmlGrowthRecorder.h"
#include "Simulator.h"
#include "Model.h"
#include "AllIFNeurons.h"      // TODO: remove LIF model specific code
#include "ConnGrowth.h"

//! THe constructor and destructor
XmlGrowthRecorder::XmlGrowthRecorder() :
      XmlRecorder(),
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

/// Compile history information in every epoch
///
/// @param[in] neurons 	The entire list of neurons.
void XmlGrowthRecorder::compileHistories(IAllVertices &neurons) {
   XmlRecorder::compileHistories(neurons);

   shared_ptr<Connections> conns = Simulator::getInstance().getModel()->getConnections();

   BGFLOAT minRadius = dynamic_cast<ConnGrowth *>(conns.get())->growthParams_.minRadius;
   VectorMatrix &rates = (*dynamic_cast<ConnGrowth *>(conns.get())->rates_);
   VectorMatrix &radii = (*dynamic_cast<ConnGrowth *>(conns.get())->radii_);

   for (int iNeuron = 0; iNeuron < Simulator::getInstance().getTotalVertices(); iNeuron++) {
      // record firing rate to history matrix
      ratesHistory_(Simulator::getInstance().getCurrentStep(), iNeuron) = rates[iNeuron];

      // Cap minimum radius size and record radii to history matrix
      // TODO: find out why we cap this here.
      if (radii[iNeuron] < minRadius)
         radii[iNeuron] = minRadius;

      // record radius to history matrix
      radiiHistory_(Simulator::getInstance().getCurrentStep(), iNeuron) = radii[iNeuron];
   }
}

/// Writes simulation results to an output destination.
///
/// @param  neurons the Neuron list to search from.
void XmlGrowthRecorder::saveSimData(const IAllVertices &neurons) {
   // create Neuron Types matrix
   VectorMatrix neuronTypes(MATRIX_TYPE, MATRIX_INIT, 1, Simulator::getInstance().getTotalVertices(), EXC);
   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      neuronTypes[i] = Simulator::getInstance().getModel()->getLayout()->neuronTypeMap_[i];
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

   // create starter nuerons matrix
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
   XmlRecorder::printParameters();

   LOG4CPLUS_DEBUG(fileLogger_, "\n---XmlGrowthRecorder Parameters---" << endl
                                      << "\tRecorder type: XmlGrowthRecorder" << endl);
}

