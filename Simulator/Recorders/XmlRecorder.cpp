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
   for(int i = 0; i < variable_first->getNumEventsInEpoch(); i++){
      //std::cout << "test output:" << (*variable_first)[i] << endl;
      single_neuron_History_.push_back((*variable_first)[i]);
   }
   variable_first->startNewEpoch();
}

/// Writes simulation results to an output destination.
/// @param  neurons the Neuron list to search from.
void XmlRecorder::saveSimData(const AllVertices &vertices)
{
   for(int i = 0; i < variable_first->getNumEventsInEpoch(); i++){
      cout << "test output:" << (*variable_first)[i] << endl;
   }
   resultOut_ << "   " << toXML(single_neuron_name, single_neuron_History_) << endl;

}

/// convert internal buffer to XML string
string XmlRecorder::toXML(string name, vector<uint64_t> single_neuron_buffer) const
{
   stringstream os;

   os << "Event for a signle neuron ";
   if (name != "")
      os << "name=\"" << name << "\" ";
   os << "type=\"complete\" rows=\"" << single_neuron_buffer.size() << "\" columns=\""
      << 2  << endl;
   for (int i = 0; i < single_neuron_buffer.size(); i++) {
      os << name << ": ";
      os << single_neuron_buffer[i] << ": ";
      os << endl;
   }
   os << endl;
   //os << "</Matrix>";

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


// print out the content in the variable table
void XmlRecorder::registerVariables(string name, EventBuffer &recordVar)
{
   // for(int i = 0; i < variableTable.size(); i++){
   //    std::cout << "name: " << variableTable[i].variableName << endl;
   //    std::cout << "location" << variableTable[i].variableLocation << endl;

   // }
   //EventBuffer &variable_first_test = recordVar;
   //std::cout << "test_output_address:" <<  variable_first_test.getNumEventsInEpoch() << endl;
   single_neuron_name = name;
   variable_first= &recordVar;
   //std::cout << "test_output2: << &variable_first << endl;

}
