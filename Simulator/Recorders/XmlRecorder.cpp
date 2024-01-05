/**
 * @file XmlRecorder.cpp
 *
 * @ingroup Simulator/Recorders
 *
 * @brief An implementation for recording variable information on xml file
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

/// Create a new xml file and initialize data
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

// TODO : @param[in] vertices will be removed eventually
/// Compile history information in every epoch
void XmlRecorder::compileHistories(AllVertices &vertices)
{
   //capture data information in each epoch
   for (int rowIndex = 0; rowIndex < variableTable_.size(); rowIndex++) {
      if (variableTable_[rowIndex].variableLocation_.getNumEventsInEpoch() > 0) {
         for (int columnIndex = 0;
              columnIndex < variableTable_[rowIndex].variableLocation_.getNumEventsInEpoch();
              columnIndex++) {
            variableTable_[rowIndex].variableHistory_.push_back(
               variableTable_[rowIndex].variableLocation_.getElement(columnIndex));
         }
      }
      variableTable_[rowIndex].variableLocation_.startNewEpoch();
   }


   // // generate the regression test files using prervious version of XmlRecorder
   // //All neurons event
   // AllSpikingNeurons &spNeurons = dynamic_cast<AllSpikingNeurons &>(vertices);
   // Simulator &simulator = Simulator::getInstance();
   // int maxSpikes = static_cast<int>(simulator.getEpochDuration() * simulator.getMaxFiringRate());

   // for (int rowIndex = 0; rowIndex < spNeurons.vertexEvents_.size(); rowIndex++) {
   //    for (int eventIterator = 0;
   //         eventIterator < spNeurons.vertexEvents_[rowIndex].getNumEventsInEpoch();
   //         eventIterator++) {
   //       variablesHistory_[rowIndex].push_back(
   //       static_cast<int>(static_cast<double>(spNeurons.vertexEvents_[rowIndex][eventIterator])));
   //    }
   // }
   // spNeurons.clearSpikeCounts();
}

// TODO : @param[in] vertices will be removed eventually
/// Writes simulation results to an output destination.
void XmlRecorder::saveSimData(const AllVertices &vertices)
{
   // Write XML header information:
   resultOut_ << "<?xml version=\"1.0\" standalone=\"no\"?>\n";
   //iterate the variable list row by row then output the cumulative value to a xml file
   for (int rowIndex = 0; rowIndex < variableTable_.size(); rowIndex++) {
      if (variableTable_[rowIndex].variableHistory_.size() > 0) {
         resultOut_ << toXML(variableTable_[rowIndex].variableName_,
                             variableTable_[rowIndex].variableHistory_,
                             variableTable_[rowIndex].dataType_)
                    << endl;
      }
   }
}

string XmlRecorder::toXML(const string &name, vector<multipleTypes> &singleBuffer_,
                          const string &basicType) const
{
   stringstream os;

   //  output file header
   os << "<Matrix ";
   if (!name.empty())
      os << "name=\"" << name << "\" ";
   os << "type=\"complete\" rows=\"" << 1 << "\" columns=\"" << singleBuffer_.size()
      << "\" multiplier=\"1.0\">" << endl;
   os << "   ";

   for (const multipleTypes &element : singleBuffer_) {
      if (basicType == "uint64_t") {
         os << get<uint64_t>(element) << " ";
      } else if (basicType == "double") {
         os << get<double>(element) << " ";
      } else if (basicType == "string") {
         os << get<string>(element) << " ";
      }
      // Add more conditions if there are additional supported data types
   }

   os << endl;
   os << "</Matrix>";

   return os.str();
}

void XmlRecorder::getStarterNeuronMatrix(VectorMatrix &matrix, const vector<bool> &starterMap)
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

/**
 * Register a single instance of a class derived from RecordableBase.
 * This method allows the XmlRecorder to obtain the updating value while the simulator runs
 * by storing the address of the registered variable.
 * @param name       The name associated with the registered variable.
 * @param recordVar  A pointer to the RecordableBase object to be registered.
 */
void XmlRecorder::registerVariable(const string &varName, RecordableBase &recordVar)
{
   // add a new variable into the table
   variableTable_.push_back(singleVariableInfo(varName, recordVar));
}

/**
 * Register a vector of instances of classes derived from RecordableBase.
 *
 * This method allows the XmlRecorder to store a vector of variables, each represented by
 * an address and a unique variable name. It is typically used to register multiple instances
 * of a class derived from RecordableBase.
 * @param varName     The name associated with the registered variables.
 * @param recordVars  A vector of pointers to RecordableBase objects to be registered.
 */
void XmlRecorder::registerVariable(const string &varName, vector<RecordableBase *> &recordVars)
{
   for (int i = 0; i < recordVars.size(); i++) {
      string variableID = varName + to_string(i);
      RecordableBase &address = *recordVars[i];
      // add a new variable into the table
      variableTable_.push_back(singleVariableInfo(variableID, address));
   }
}
