 /*
 *      @file XmlGrowthRecorder.cpp
 *
 *      @brief An implementation for recording spikes history on xml file
 */
//! An implementation for recording spikes history on xml file

/**
 @file XmlSTDPRecorder.cpp
 @brief An implementation for recording weights of synapses for STDP on xml file
 @author Snigdha Singh
 @date January 2021
 @version 1
 */

#include "XmlSTDPRecorder.h"
#include "Simulator.h"
#include "Model.h"
#include "AllIFNeurons.h"      // TODO: remove LIF model specific code
#include "ConnStatic.h"
#include <vector>

//! THe constructor and destructor
XmlSTDPRecorder::XmlSTDPRecorder() :
XmlRecorder()
     /*
      weightsHistory_(MATRIX_TYPE, MATRIX_INIT, static_cast<int>(Simulator::getInstance().getNumEpochs() + 1),
                    Simulator::getInstance().getTotalNeurons()*Simulator::getInstance().getMaxSynapsesPerNeuron()),
      sourceNeuronsHistory_(MATRIX_TYPE, MATRIX_INIT, static_cast<int>(Simulator::getInstance().getNumEpochs() + 1),
                    Simulator::getInstance().getTotalNeurons()*Simulator::getInstance().getMaxSynapsesPerNeuron()),  

      destNeuronsHistory_(MATRIX_TYPE, MATRIX_INIT, static_cast<int>(Simulator::getInstance().getNumEpochs() + 1),
                    Simulator::getInstance().getTotalNeurons()*Simulator::getInstance().getMaxSynapsesPerNeuron()) 
                    */
                   
      {
   
  weightsHistory_.resize( Simulator::getInstance().getNumEpochs() + 1 , 
      vector<BGFLOAT> (Simulator::getInstance().getTotalNeurons()*Simulator::getInstance().getMaxSynapsesPerNeuron(),0));
   
    sourceNeuronIndexHistory_.resize( Simulator::getInstance().getNumEpochs() + 1 , 
      vector<int> (Simulator::getInstance().getTotalNeurons()*Simulator::getInstance().getMaxSynapsesPerNeuron()));

destNeuronIndexHistory_.resize( Simulator::getInstance().getNumEpochs() + 1 , 
      vector<int> (Simulator::getInstance().getTotalNeurons()*Simulator::getInstance().getMaxSynapsesPerNeuron()));
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


}

/*
 * Init radii and rates history matrices with current radii and rates
 */
void XmlSTDPRecorder::initValues() {
   shared_ptr<Connections> conns = Simulator::getInstance().getModel()->getConnections();

/*
   for (int i = 0; i < Simulator::getInstance().getTotalNeurons()*Simulator::getInstance().getMaxSynapsesPerNeuron(); i++) {
      weightsHistory_(0, i) = (dynamic_cast<ConnStatic *>(conns.get())->WCurrentEpoch_)[i];
      sourceNeuronsHistory_(0, i) = (dynamic_cast<ConnStatic *>(conns.get())->sourceNeuronIndexCurrentEpoch_)[i];
      destNeuronsHistory_(0, i) = (dynamic_cast<ConnStatic *>(conns.get())->destNeuronIndexCurrentEpoch_)[i];
      
   }*/
}

/*
 * Get the current radii and rates values
 */
void XmlSTDPRecorder::getValues() {
   Connections *conns = Simulator::getInstance().getModel()->getConnections().get();

   for (int i = 0; i < Simulator::getInstance().getTotalNeurons(); i++) {
      (dynamic_cast<ConnStatic *>(conns)->WCurrentEpoch_)[i] = weightsHistory_[Simulator::getInstance().getCurrentStep()][i];
      (dynamic_cast<ConnStatic *>(conns)->sourceNeuronIndexCurrentEpoch_)[i] = sourceNeuronIndexHistory_[Simulator::getInstance().getCurrentStep()][i];
      (dynamic_cast<ConnStatic *>(conns)->destNeuronIndexCurrentEpoch_)[i] = destNeuronIndexHistory_[Simulator::getInstance().getCurrentStep()][i];
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
   //TO DO: change to int
   int &sourceIndex = (*dynamic_cast<ConnStatic *>(conns.get())->sourceNeuronIndexCurrentEpoch_);
   int &destIndex = (*dynamic_cast<ConnStatic *>(conns.get())->destNeuronIndexCurrentEpoch_);
    
      

   for (int iNeuron = 0; iNeuron < Simulator::getInstance().getTotalNeurons()*Simulator::getInstance().getMaxSynapsesPerNeuron(); iNeuron++) {
      // record firing rate to history matrix

      weightsHistory_[Simulator::getInstance().getCurrentStep()][iNeuron]= (dynamic_cast<ConnStatic *>(conns.get())->WCurrentEpoch_)[iNeuron];
      sourceNeuronIndexHistory_[Simulator::getInstance().getCurrentStep()][iNeuron]= (dynamic_cast<ConnStatic *>(conns.get())->sourceNeuronIndexCurrentEpoch_)[iNeuron];
      destNeuronIndexHistory_[Simulator::getInstance().getCurrentStep()][iNeuron]= (dynamic_cast<ConnStatic *>(conns.get())->destNeuronIndexCurrentEpoch_)[iNeuron];
      //LOG4CPLUS_INFO(fileLogger_, Simulator::getInstance().getCurrentStep()<<" "<< iNeuron);
      //weightsHistory_(Simulator::getInstance().getCurrentStep(), iNeuron) = weights[iNeuron];
   }
   LOG4CPLUS_INFO(fileLogger_, "Finished Compiling STDP HISTORY");
}

// convert Matrix to XML string
string XmlSTDPRecorder::toXML(string name,vector<vector<BGFLOAT>> MatrixToWrite) const
{
    stringstream os;
    
    os << "<Matrix ";
    if (name != "")
        os << "name=\"" << name << "\" ";
    os << "type=\"complete\" rows=\"" << MatrixToWrite.size()
    << "\" columns=\"" << MatrixToWrite[0].size()
    << "\" multiplier=\"1.0\">" << endl;
    for (int i = 0; i < MatrixToWrite.size(); i++)
{
    for (int j = 0; j < MatrixToWrite[i].size(); j++)
    {
        os << MatrixToWrite[i][j]<< " ";
    }
}
os <<endl;
    os << "</Matrix>";
    
    return os.str();
}

// convert Matrix to XML string
string XmlSTDPRecorder::toXML(string name,vector<vector<int>> MatrixToWrite) const
{
    stringstream os;
    
    os << "<Matrix ";
    if (name != "")
        os << "name=\"" << name << "\" ";
    os << "type=\"complete\" rows=\"" << MatrixToWrite.size()
    << "\" columns=\"" << MatrixToWrite[0].size()
    << "\" multiplier=\"1.0\">" << endl;
    for (int i = 0; i < MatrixToWrite.size(); i++)
{
    for (int j = 0; j < MatrixToWrite[i].size(); j++)
    {
        os << MatrixToWrite[i][j]<<" ";
    }
}
os <<endl;
    os << "</Matrix>";
    
    return os.str();
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
   stateOut_ << "   " <<toXML("sourceNeuronIndexHistory",sourceNeuronIndexHistory_) << endl;
   stateOut_ << "   " << toXML("destNeuronIndexHistory",destNeuronIndexHistory_) << endl;
   stateOut_ << "   " <<toXML("weightsHistory",weightsHistory_) << endl;
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

