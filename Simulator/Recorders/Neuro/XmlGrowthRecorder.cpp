/**
 * @file XmlGrowthRecorder.cpp
 *
 * @ingroup Simulator/Recorders
 *
 * @brief An implementation for recording spikes history in an XML file for growth simulations
 */

#include "XmlGrowthRecorder.h"
#include "AllIFNeurons.h"   // TODO: remove LIF model specific code
#include "ConnGrowth.h"
#include "Model.h"
#include "Simulator.h"

// TODO: We don't need the explicit call to the superclass constructor, right?
//! The constructor and destructor
XmlGrowthRecorder::XmlGrowthRecorder()
{
}

// TODO: Is this needed?
XmlGrowthRecorder::~XmlGrowthRecorder()
{
}

void XmlGrowthRecorder::init()
{
   // call the superclass method first
   XmlRecorder::init();

   // Allocate memory for ratesHistory and radiiHistory
   ratesHistory_ = shared_ptr<CompleteMatrix>(new CompleteMatrix(
      MATRIX_TYPE, MATRIX_INIT, static_cast<int>(Simulator::getInstance().getNumEpochs() + 1),
      Simulator::getInstance().getTotalVertices()));

   radiiHistory_ = shared_ptr<CompleteMatrix>(new CompleteMatrix(
      MATRIX_TYPE, MATRIX_INIT, static_cast<int>(Simulator::getInstance().getNumEpochs() + 1),
      Simulator::getInstance().getTotalVertices()));
}

/// Init radii and rates history matrices with default values
void XmlGrowthRecorder::initDefaultValues()
{
   shared_ptr<Connections> conns = Simulator::getInstance().getModel()->getConnections();
   BGFLOAT startRadius = dynamic_cast<ConnGrowth *>(conns.get())->growthParams_.startRadius;

   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      (*radiiHistory_)(0, i) = startRadius;
      (*ratesHistory_)(0, i) = 0;
   }
}

/// Init radii and rates history matrices with current radii and rates
void XmlGrowthRecorder::initValues()
{
   shared_ptr<Connections> conns = Simulator::getInstance().getModel()->getConnections();

   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      (*radiiHistory_)(0, i) = (*dynamic_cast<ConnGrowth *>(conns.get())->radii_)[i];
      (*ratesHistory_)(0, i) = (*dynamic_cast<ConnGrowth *>(conns.get())->rates_)[i];
   }
}

/// Get the current radii and rates values
void XmlGrowthRecorder::getValues()
{
   Connections *conns = Simulator::getInstance().getModel()->getConnections().get();

   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      (*dynamic_cast<ConnGrowth *>(conns)->radii_)[i]
         = (*radiiHistory_)(Simulator::getInstance().getCurrentStep(), i);
      (*dynamic_cast<ConnGrowth *>(conns)->rates_)[i]
         = (*ratesHistory_)(Simulator::getInstance().getCurrentStep(), i);
   }
}

/// Compile history information in every epoch
///
/// @param[in] neurons 	The entire list of neurons.
void XmlGrowthRecorder::compileHistories(AllVertices &neurons)
{
   XmlRecorder::compileHistories(neurons);

   shared_ptr<Connections> conns = Simulator::getInstance().getModel()->getConnections();

   BGFLOAT minRadius = dynamic_cast<ConnGrowth *>(conns.get())->growthParams_.minRadius;
   VectorMatrix &rates = (*dynamic_cast<ConnGrowth *>(conns.get())->rates_);
   VectorMatrix &radii = (*dynamic_cast<ConnGrowth *>(conns.get())->radii_);

   for (int iVertex = 0; iVertex < Simulator::getInstance().getTotalVertices(); iVertex++) {
      // record firing rate to history matrix
      (*ratesHistory_)(Simulator::getInstance().getCurrentStep(), iVertex) = rates[iVertex];

      // Cap minimum radius size and record radii to history matrix
      // TODO: find out why we cap this here.
      // TODO: agreed; seems like this should be capped elsewhere.
      if (radii[iVertex] < minRadius)
         radii[iVertex] = minRadius;

      // record radius to history matrix
      (*radiiHistory_)(Simulator::getInstance().getCurrentStep(), iVertex) = radii[iVertex];
   }
}


/// Writes simulation results to an output destination.
///
/// @param  neurons the Neuron list to search from.
void XmlGrowthRecorder::saveSimData(const AllVertices &neurons)
{
   // create Neuron Types matrix
   VectorMatrix neuronTypes(MATRIX_TYPE, MATRIX_INIT, 1,
                            Simulator::getInstance().getTotalVertices(), EXC);
   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      neuronTypes[i] = Simulator::getInstance().getModel()->getLayout()->vertexTypeMap_[i];
   }

   // create neuron threshold matrix
   VectorMatrix neuronThresh(MATRIX_TYPE, MATRIX_INIT, 1,
                             Simulator::getInstance().getTotalVertices(), 0);
   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      neuronThresh[i] = dynamic_cast<const AllIFNeurons &>(neurons).Vthresh_[i];
   }

   // Write XML header information:
   resultOut_ << "<?xml version=\"1.0\" standalone=\"no\"?>\n"
              << "<!-- State output file for the DCT growth modeling-->\n";
   //stateOut << version; TODO: version

   // Write the core state information:
   resultOut_ << "<SimState>\n";
   resultOut_ << "   " << radiiHistory_->toXML("radiiHistory") << endl;
   resultOut_ << "   " << ratesHistory_->toXML("ratesHistory") << endl;
   resultOut_ << "   " << spikesHistory_.toXML("spikesHistory") << endl;
   resultOut_ << "   " << Simulator::getInstance().getModel()->getLayout()->xloc_->toXML("xloc")
              << endl;
   resultOut_ << "   " << Simulator::getInstance().getModel()->getLayout()->yloc_->toXML("yloc")
              << endl;
   resultOut_ << "   " << neuronTypes.toXML("neuronTypes") << endl;

   // create starter neuron matrix
   int num_starter_neurons = static_cast<int>(
      Simulator::getInstance().getModel()->getLayout()->numEndogenouslyActiveNeurons_);
   if (num_starter_neurons > 0) {
      VectorMatrix starterNeurons(MATRIX_TYPE, MATRIX_INIT, 1, num_starter_neurons);
      getStarterNeuronMatrix(starterNeurons,
                             Simulator::getInstance().getModel()->getLayout()->starterMap_);
      resultOut_ << "   " << starterNeurons.toXML("starterNeurons") << endl;
   }

   // Write neuron thresold
   resultOut_ << "   " << neuronThresh.toXML("neuronThresh") << endl;

   // write time between growth cycles
   resultOut_
      << "   <Matrix name=\"Tsim\" type=\"complete\" rows=\"1\" columns=\"1\" multiplier=\"1.0\">"
      << endl;
   resultOut_ << "   " << Simulator::getInstance().getEpochDuration() << endl;
   resultOut_ << "</Matrix>" << endl;

   // write simulation end time
   resultOut_
      << "   <Matrix name=\"simulationEndTime\" type=\"complete\" rows=\"1\" columns=\"1\" multiplier=\"1.0\">"
      << endl;
   resultOut_ << "   " << g_simulationStep * Simulator::getInstance().getDeltaT() << endl;
   resultOut_ << "</Matrix>" << endl;
   resultOut_ << "</SimState>" << endl;
}

///  Prints out all parameters to logging file.
///  Registered to OperationManager as Operation::printParameters
void XmlGrowthRecorder::printParameters()
{
   LOG4CPLUS_DEBUG(fileLogger_, "\nXMLRECORDER PARAMETERS"
                                   << endl
                                   << "\tResult file path: " << resultFileName_ << endl
                                   << "\tSpikes History Size: " << spikesHistory_.Size() << endl
                                   << "\n---XmlGrowthRecorder Parameters---" << endl
                                   << "\tRecorder type: XmlGrowthRecorder" << endl);
}

///  Get starter Neuron matrix.
///
///  @param  matrix      Starter Neuron matrix.
///  @param  starterMap Bool map to reference neuron matrix location from.
void XmlGrowthRecorder::getStarterNeuronMatrix(VectorMatrix &matrix, const bool *starterMap)
{
   int cur = 0;
   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      if (starterMap[i]) {
         matrix[cur] = i;
         cur++;
      }
   }
}
