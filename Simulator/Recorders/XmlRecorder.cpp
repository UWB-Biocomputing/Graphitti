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
   for (int i = 0; i < singleNeuronEvents_->getNumEventsInEpoch(); i++) {
      //std::cout << "test output:" << (*variable_first)[i] << endl;
      single_neuron_History_.push_back((*singleNeuronEvents_)[i]);
   }
   singleNeuronEvents_->startNewEpoch();
}

/// Writes simulation results to an output destination.
/// @param  neurons the Neuron list to search from.
void XmlRecorder::saveSimData(const AllVertices &vertices)
{
   // Write XML header information:
   resultOut_ << "<?xml version=\"1.0\" standalone=\"no\"?>\n"
              << "<!-- State output file for the DCT growth modeling-->\n";
   resultOut_ << "   " << toXML(neuronName, single_neuron_History_) << endl;
}

/// convert internal buffer to XML string
string XmlRecorder::toXML(string name, vector<uint64_t> single_neuron_buffer) const
{
   stringstream os;

   os << "<Matrix ";
   if (name != "")
      os << "name=\"" << name << "\" ";
   os << "type=\"complete\" rows=\"" << 1 << "\" columns=\""
      << single_neuron_buffer.size() << "\" multiplier=\"1.0\">" << endl;
   os << "   ";
   for (int i = 0; i < single_neuron_buffer.size(); i++) {
      os << single_neuron_buffer[i] << " ";
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


/// Store the neuron number and all the events for this neuron that registered in the variable owner class
void XmlRecorder::registerVariables(string name, EventBuffer &recordVar)
{
   neuronName = name;
   singleNeuronEvents_ = &recordVar;
}
