 /*
 *      @file XmlSTDPRecorder.cpp
 *      @brief An implementation for recording weights of synapses for STDP on xml file
 *      @author Snigdha Singh
 *       @date January 2021
 */


#include "XmlSTDPRecorder.h"
#include "Simulator.h"
#include "Model.h"
#include "AllIFNeurons.h"      // TODO: remove LIF model specific code
#include "ConnStatic.h"
#include <vector>

// TODO: We don't need to explicitly call the superclass constructor, right?
//! The constructor and destructor
XmlSTDPRecorder::XmlSTDPRecorder() :
XmlRecorder()
{
   const int numEpochs = Simulator::getInstance().getNumEpochs();
   const int totalNeurons = Simulator::getInstance().getTotalVertices();
   const int maxSynapsesPerNeuron = Simulator::getInstance().getMaxEdgesPerVertex();
   
   weightsHistory_.resize(numEpochs + 1,
      vector<BGFLOAT> (totalNeurons*maxSynapsesPerNeuron,0));
   
   sourceNeuronIndexHistory_.resize(numEpochs + 1,
      vector<int> (totalNeurons*maxSynapsesPerNeuron));

   destNeuronIndexHistory_.resize(numEpochs + 1 ,
      vector<int> (totalNeurons*maxSynapsesPerNeuron));
}

XmlSTDPRecorder::~XmlSTDPRecorder() {
}

///Init radii and rates history matrices with default values
void XmlSTDPRecorder::initDefaultValues() {
   shared_ptr<Connections> conns = Simulator::getInstance().getModel()->getConnections();
  //AllNeuroEdges *synapses synapses)->W_[iSyn]
   BGFLOAT startRadius = dynamic_cast<ConnStatic *>(conns.get())->getConnsRadiusThresh();


}

///InitValues gets the values for weights, source index and dest index at the time of simulation start
void XmlSTDPRecorder::initValues() {
   Connections *conns = Simulator::getInstance().getModel()->getConnections().get();


   for (int i = 0; i < Simulator::getInstance().getTotalVertices()*Simulator::getInstance().getMaxEdgesPerVertex(); i++) {
      weightsHistory_[0][i] = (dynamic_cast<ConnStatic *>(conns)->getWCurrentEpoch())[i];
      sourceNeuronIndexHistory_[0][i] = (dynamic_cast<ConnStatic *>(conns)->getSourceVertexIndexCurrentEpoch())[i];
      destNeuronIndexHistory_[0][i] = (dynamic_cast<ConnStatic *>(conns)->getDestVertexIndexCurrentEpoch())[i];
      
   }
}

/// Get the current synapse weight information
void XmlSTDPRecorder::getValues() {
   Simulator &simulator = Simulator::getInstance();
   shared_ptr<Connections> connections = simulator.getModel()->getConnections();
   AllEdges &synapses =(*connections->getEdges());
   const int currentStep = simulator.getCurrentStep();

   for (int i = 0; i < simulator.getTotalVertices(); i++) {
      synapses.W_[i] = weightsHistory_[currentStep][i];
      synapses.sourceVertexIndex_[i] = sourceNeuronIndexHistory_[currentStep][i];
      synapses.destVertexIndex_[i] = destNeuronIndexHistory_[currentStep][i];
   }
}

/// Compile history information in every epoch
///
/// @param[in] neurons 	The entire list of neurons.
void XmlSTDPRecorder::compileHistories(AllVertices &neurons) {
   LOG4CPLUS_INFO(fileLogger_, "Compiling STDP HISTORY");
   XmlRecorder::compileHistories(neurons);
   Simulator &simulator = Simulator::getInstance();
   shared_ptr<Connections> connections = simulator.getModel()->getConnections();
   AllEdges &synapses =(*connections->getEdges());
   const int currentStep = simulator.getCurrentStep();

   for (int iNeuron = 0; iNeuron < simulator.getTotalVertices()*simulator.getMaxEdgesPerVertex(); iNeuron++) {
      // record firing rate to history matrix
      weightsHistory_[currentStep][iNeuron]= synapses.W_[iNeuron];
      sourceNeuronIndexHistory_[currentStep][iNeuron]= synapses.sourceVertexIndex_[iNeuron];
      destNeuronIndexHistory_[currentStep][iNeuron]= synapses.destVertexIndex_[iNeuron];
   }
   LOG4CPLUS_INFO(fileLogger_, "Finished Compiling STDP HISTORY");
}

/// convert Matrix to XML string
string XmlSTDPRecorder::toXML(string name, vector<vector<BGFLOAT>> MatrixToWrite) const
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
     os<<endl;
}
os <<endl;
    os << "</Matrix>";
    
    return os.str();
}

/// convert Matrix to XML string
string XmlSTDPRecorder::toXML(string name, vector<vector<int>> MatrixToWrite) const
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
    os<<endl;
}
   os <<endl;
    os << "</Matrix>";
    
    return os.str();
}

/// Writes simulation results to an output destination.
///
/// @param  neurons the Neuron list to search from.
void XmlSTDPRecorder::saveSimData(const AllVertices &neurons)
{
   Simulator &simulator = Simulator::getInstance();
   
   // create Neuron Types matrix
   VectorMatrix neuronTypes(MATRIX_TYPE, MATRIX_INIT, 1, simulator.getTotalVertices(), EXC);
   for (int i = 0; i < simulator.getTotalVertices(); i++) {
      neuronTypes[i] = simulator.getModel()->getLayout()->vertexTypeMap_[i];
   }

   // create neuron threshold matrix
   VectorMatrix neuronThresh(MATRIX_TYPE, MATRIX_INIT, 1, simulator.getTotalVertices(), 0);
   for (int i = 0; i < simulator.getTotalVertices(); i++) {
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
   stateOut_ << "   " << simulator.getModel()->getLayout()->xloc_->toXML("xloc") << endl;
   stateOut_ << "   " << simulator.getModel()->getLayout()->yloc_->toXML("yloc") << endl;
   stateOut_ << "   " << neuronTypes.toXML("neuronTypes") << endl;

   // create starter neuron matrix
   int num_starter_neurons = static_cast<int>(simulator.getModel()->getLayout()->numEndogenouslyActiveNeurons_);
   if (num_starter_neurons > 0) {
      VectorMatrix starterNeurons(MATRIX_TYPE, MATRIX_INIT, 1, num_starter_neurons);
      getStarterNeuronMatrix(starterNeurons, simulator.getModel()->getLayout()->starterMap_);
      stateOut_ << "   " << starterNeurons.toXML("starterNeurons") << endl;
   }

   // Write neuron thresold
   stateOut_ << "   " << neuronThresh.toXML("neuronThresh") << endl;

   // write time between growth cycles
   stateOut_ << "   <Matrix name=\"Tsim\" type=\"complete\" rows=\"1\" columns=\"1\" multiplier=\"1.0\">" << endl;
   stateOut_ << "   " << simulator.getEpochDuration() << endl;
   stateOut_ << "</Matrix>" << endl;

   // write simulation end time
   stateOut_ << "   <Matrix name=\"simulationEndTime\" type=\"complete\" rows=\"1\" columns=\"1\" multiplier=\"1.0\">"
             << endl;
   stateOut_ << "   " << g_simulationStep * simulator.getDeltaT() << endl;
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

