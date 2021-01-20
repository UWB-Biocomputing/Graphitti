 /*
 *      @file XmlGrowthRecorder.cpp
 *
 *      @brief An implementation for recording spikes history on xml file
 */
//! An implementation for recording spikes history on xml file

#include "XmlSTDPRecorder.h"
#include "Simulator.h"
#include "Model.h"
#include "AllIFNeurons.h"      // TODO: remove LIF model specific code
#include "ConnStatic.h"

//! THe constructor and destructor
XmlSTDPRecorder::XmlSTDPRecorder() :
      XmlRecorder(),
      weightsHistory_(MATRIX_TYPE, MATRIX_INIT, static_cast<int>(Simulator::getInstance().getNumEpochs() + 1),
                    Simulator::getInstance().getTotalNeurons()*Simulator::getInstance().getMaxSynapsesPerNeuron()),
      sourceNeuronsHistory_(MATRIX_TYPE, MATRIX_INIT, static_cast<int>(Simulator::getInstance().getNumEpochs() + 1),
                    Simulator::getInstance().getTotalNeurons()*Simulator::getInstance().getMaxSynapsesPerNeuron()),  

      destNeuronsHistory_(MATRIX_TYPE, MATRIX_INIT, static_cast<int>(Simulator::getInstance().getNumEpochs() + 1),
                    Simulator::getInstance().getTotalNeurons()*Simulator::getInstance().getMaxSynapsesPerNeuron()) {

}
XmlSTDPRecorder::~XmlSTDPRecorder() {
}

/*
 * Init radii and rates history matrices with default values
 */
void XmlSTDPRecorder::initDefaultValues() {
   shared_ptr<Connections> conns = Simulator::getInstance().getModel()->getConnections();
  //IAllSynapses *synapses synapses)->W_[iSyn]
   BGFLOAT startRadius = dynamic_cast<ConnStatic *>(conns.get())->threshConnsRadius_;

   for (int i = 0; i < Simulator::getInstance().getTotalNeurons(); i++) {
      //radiiHistory_(0, i) = startRadius;
      //ratesHistory_(0, i) = 0;
   }
}

/*
 * Init radii and rates history matrices with current radii and rates
 */
void XmlSTDPRecorder::initValues() {
   shared_ptr<Connections> conns = Simulator::getInstance().getModel()->getConnections();

   for (int i = 0; i < Simulator::getInstance().getTotalNeurons()*Simulator::getInstance().getMaxSynapsesPerNeuron(); i++) {
      weightsHistory_(0, i) = (dynamic_cast<ConnStatic *>(conns.get())->WCurrentEpoch_)[i];
      sourceNeuronsHistory_(0, i) = (dynamic_cast<ConnStatic *>(conns.get())->sourceNeuronIndexCurrentEpoch_)[i];
      destNeuronsHistory_(0, i) = (dynamic_cast<ConnStatic *>(conns.get())->destNeuronIndexCurrentEpoch_)[i];
      
   }
}

/*
 * Get the current radii and rates values
 */
void XmlSTDPRecorder::getValues() {
   Connections *conns = Simulator::getInstance().getModel()->getConnections().get();

   for (int i = 0; i < Simulator::getInstance().getTotalNeurons(); i++) {
      (dynamic_cast<ConnStatic *>(conns)->WCurrentEpoch_)[i] = weightsHistory_(Simulator::getInstance().getCurrentStep(), i);
      (dynamic_cast<ConnStatic *>(conns)->sourceNeuronIndexCurrentEpoch_)[i] = sourceNeuronsHistory_(Simulator::getInstance().getCurrentStep(), i);
      (dynamic_cast<ConnStatic *>(conns)->destNeuronIndexCurrentEpoch_)[i] = destNeuronsHistory_(Simulator::getInstance().getCurrentStep(), i);
      //(*dynamic_cast<ConnGrowth *>(conns)->rates_)[i] = ratesHistory_(Simulator::getInstance().getCurrentStep(), i);
   }
}

/*
 * Compile history information in every epoch
 *
 * @param[in] neurons 	The entire list of neurons.
 */
void XmlSTDPRecorder::compileHistories(IAllNeurons &neurons) {
   LOG4CPLUS_INFO(fileLogger_, "Compiling STDP HISTORY");
   XmlRecorder::compileHistories(neurons);
   //LOG4CPLUS_INFO(fileLogger_, "Compiling STDP HISTORY");
   shared_ptr<Connections> conns = Simulator::getInstance().getModel()->getConnections();

   //VectorMatrix &rates = (*dynamic_cast<ConnGrowth *>(conns.get())->rates_);
   BGFLOAT &weights = (*dynamic_cast<ConnStatic *>(conns.get())->WCurrentEpoch_);
   BGFLOAT &sourceIndex = (*dynamic_cast<ConnStatic *>(conns.get())->sourceNeuronIndexCurrentEpoch_);
   BGFLOAT &destIndex = (*dynamic_cast<ConnStatic *>(conns.get())->destNeuronIndexCurrentEpoch_);
  //sourceNeuronIndexCurrentEpoch_


   for (int iNeuron = 0; iNeuron < Simulator::getInstance().getTotalNeurons()*Simulator::getInstance().getMaxSynapsesPerNeuron(); iNeuron++) {
      // record firing rate to history matrix
      
      weightsHistory_(Simulator::getInstance().getCurrentStep(), iNeuron)= (dynamic_cast<ConnStatic *>(conns.get())->WCurrentEpoch_)[iNeuron];
      sourceNeuronsHistory_(Simulator::getInstance().getCurrentStep(), iNeuron)= (dynamic_cast<ConnStatic *>(conns.get())->sourceNeuronIndexCurrentEpoch_)[iNeuron];
      destNeuronsHistory_(Simulator::getInstance().getCurrentStep(), iNeuron)= (dynamic_cast<ConnStatic *>(conns.get())->destNeuronIndexCurrentEpoch_)[iNeuron];
      //LOG4CPLUS_INFO(fileLogger_, Simulator::getInstance().getCurrentStep()<<" "<< iNeuron);
      //weightsHistory_(Simulator::getInstance().getCurrentStep(), iNeuron) = weights[iNeuron];
   }
   LOG4CPLUS_INFO(fileLogger_, "Finished Compiling STDP HISTORY");
}

/*
 * Writes simulation results to an output destination.
 *
 * @param  neurons the Neuron list to search from.
 **/
void XmlSTDPRecorder::saveSimData(const IAllNeurons &neurons) {
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

   // Write the core state information:
   stateOut_ << "<SimState>\n";
   stateOut_ << "   " << sourceNeuronsHistory_.toXML("sourceNeuronIndexHistory") << endl;
   stateOut_ << "   " << destNeuronsHistory_.toXML("destNeuronHistory") << endl;
   stateOut_ << "   " << weightsHistory_.toXML("weightsHistory") << endl;
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

/**
 *  Prints out all parameters to logging file.
 *  Registered to OperationManager as Operation::printParameters
 */
void XmlSTDPRecorder::printParameters() {
   XmlRecorder::printParameters();

   LOG4CPLUS_DEBUG(fileLogger_, "\n---XmlSTDPRecorder Parameters---" << endl
                                      << "\tRecorder type: XmlSTDPRecorder" << endl);
}

