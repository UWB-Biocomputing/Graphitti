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
XmlRecorder::XmlRecorder()
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
   // check the output file extension is .xml
   string suffix = ".xml";
   if ((resultFileName_.size() <= suffix.size())
       || (resultFileName_.compare(resultFileName_.size() - suffix.size(), suffix.size(), suffix)
           != 0)) {
      perror("the file extention is not .xml ");
      exit(EXIT_FAILURE);
   }

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
   // for(int i = 0; i < variableTable.size(); i++){
   //    if (variableTable[i].variableLocation_->getNumEventsInEpoch() <= 0) {
   //       cout << "empty: " << variableTable[i].variableName_<< endl;
   //    }else {
   //       cout << variableTable[i].variableName_<< endl;
   //       for (int j = 0; j < variableTable[i].variableLocation_->getNumEventsInEpoch(); j++) {
   //          // std::cout << "test output:" << (*(variableTable[i].variableLocation_))[j] << endl;
   //          //singleNeuronHistory_.push_back((*singleNeuronEvents_)[i]);
   //          cout << (*(variableTable[i].variableLocation_))[j] << " ";
   //          neuronsHistory_[i].push_back((*(variableTable[i].variableLocation_))[j]);
   //       }
   //       cout << endl;
   //       variableTable[i].variableLocation_->startNewEpoch();

   //    }

   // }
   // for (int i = 0; i < singleNeuronEvents_->getNumEventsInEpoch(); i++) {
   //    //std::cout << "test output:" << (**singleNeuronEvents_)[i] << endl;
   //    singleNeuronHistory_.push_back((*singleNeuronEvents_)[i]);
   // }
   // singleNeuronEvents_->startNewEpoch();

   // // generate the regression test files using prervious version of XmlRecorder
   // //All neurons event
   AllSpikingNeurons &spNeurons = dynamic_cast<AllSpikingNeurons &>(vertices);
   Simulator &simulator = Simulator::getInstance();
   int maxSpikes = static_cast<int>(simulator.getEpochDuration() * simulator.getMaxFiringRate());

   for (int iNeuron = 0; iNeuron < spNeurons.vertexEvents_.size(); iNeuron++) {
      for (int eventIterator = 0;
           eventIterator < spNeurons.vertexEvents_[iNeuron].getNumEventsInEpoch();
           eventIterator++) {
         neuronsHistory_[iNeuron].push_back(
         static_cast<int>(static_cast<double>(spNeurons.vertexEvents_[iNeuron][eventIterator])));
      }
   }
   spNeurons.clearSpikeCounts();
}

/// Writes simulation results to an output destination.
/// @param  neurons the Neuron list to search from.
void XmlRecorder::saveSimData(const AllVertices &vertices)
{
   // Write XML header information:
   resultOut_ << "<?xml version=\"1.0\" standalone=\"no\"?>\n";
   // if (singleNeuronHistory_.size() != 0) {
   //    resultOut_ << toXML(neuronName_, singleNeuronHistory_) << endl;
   // }
   for(int i = 0; i < variableTable.size(); i++){
      // if (variableTable[i].variableLocation_ != nullptr) {
      if (neuronsHistory_[i].size() > 0) {
         resultOut_ << toXML(variableTable[i].variableName_, neuronsHistory_[i]) << endl;
      }

   }
}

/// Convert internal buffer to XML string
string XmlRecorder::toXML(string name, vector<uint64_t> singleNeuronBuffer_) const
{
   stringstream os;

   os << "<Matrix ";
   if (name != "")
      os << "name=\"" << name << "\" ";
   os << "type=\"complete\" rows=\"" << 1 << "\" columns=\"" << singleNeuronBuffer_.size()
      << "\" multiplier=\"1.0\">" << endl;
   os << "   ";
   for (int i = 0; i < singleNeuronBuffer_.size(); i++) {
      os << singleNeuronBuffer_[i] << " ";
   }
   os << endl;
   os << "</Matrix>";

   return os.str();
}

void XmlRecorder::getStarterNeuronMatrix(VectorMatrix &matrix, const std::vector<bool> &starterMap)
{
}

/**
 *  Prints out all parameters to logging file.
 *  Registered to OperationManager as Operation::printParameters
 */
void XmlRecorder::printParameters()
{
   LOG4CPLUS_DEBUG(fileLogger_, "\nXMLRECORDER PARAMETERS"
                                   << endl
                                   << "\tResult file path: " << resultFileName_ << endl);
}


/// Obtain the updating value while the simulator runs by storing the address of registered variable
/// Store a single neuron with the neuron number and its corresponding events
void XmlRecorder::registerVariable(string name, EventBuffer &recordVar)
{
   // neuronName_ = name;
   // singleNeuronEvents_ = std::shared_ptr<EventBuffer>(&recordVar, [](EventBuffer *) {
   // });
   // cout << name << endl;
   // cout << ": smart pointer singleNeuronEvents_" << endl;
   // cout << singleNeuronEvents_.get() << endl;
   // if(singleNeuronEvents_.get() == nullptr){
   //    cout << "Empty smart pointer" << endl;
   // }
   variableTable.push_back(variableInfo(name, recordVar));
   int newNeuron = variableTable.size() -1;
   
   if (variableTable[newNeuron].variableLocation_ != nullptr) {
      // cout << "empty: " << variableTable[i].variableName_<< endl;
      // return;
      std::vector<uint64_t> singleHistory_;
      neuronsHistory_.push_back(singleHistory_);
   }

}
