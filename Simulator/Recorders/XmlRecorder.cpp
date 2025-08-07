/**
 * @file XmlRecorder.cpp
 *
 * @ingroup Simulator/Recorders
 *
 * @brief An implementation for recording variable information on xml file
 */

#include "XmlRecorder.h"
#include "AllIFNeurons.h"        // TODO: remove LIF model specific code
#include "AllSpikingNeurons.h"   //TODO: remove after HDF5Recorder implementing
#include "ConnGrowth.h"
#include "OperationManager.h"
#include "ParameterManager.h"
#include "VectorMatrix.h"   ////TODO: remove after HDF5Recorder implementing
#include <functional>

// constructor
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

/// Terminate process
void XmlRecorder::term()
{
   resultOut_.close();
}

// TODO : @param[in] vertices will be removed eventually after HDF5Recorder implementing
/// Compile history information in every epoch
void XmlRecorder::compileHistories()
{
   //capture data information in each epoch
   for (int rowIndex = 0; rowIndex < variableTable_.size(); rowIndex++) {
      if (variableTable_[rowIndex].variableType_ == UpdatedType::DYNAMIC) {
         variableTable_[rowIndex].captureData();
         variableTable_[rowIndex].variableLocation_.startNewEpoch();
      }
   }
}

// TODO : @param[in] vertices will be removed eventually after HDF5Recorder implementing
/// Writes simulation results to an output destination.
void XmlRecorder::saveSimData()
{
   // Write XML header information:
   string header = "<?xml version=\"1.0\" standalone=\"no\"?>\n";
   resultOut_ << header;
   // Iterates the variable table to
   // (1)cpature values of Constant variable
   // (2) output the cumulative value to a xml file
   for (int rowIndex = 0; rowIndex < variableTable_.size(); rowIndex++) {
      if (variableTable_[rowIndex].variableType_ == UpdatedType::CONSTANT) {
         variableTable_[rowIndex].captureData();
      }
      // cout << variableTable_[rowIndex].variableName_ << endl;
      if (variableTable_[rowIndex].variableHistory_.size() > 0) {
         resultOut_ << toXML(variableTable_[rowIndex].variableName_,
                             variableTable_[rowIndex].variableHistory_,
                             variableTable_[rowIndex].dataType_)
                    << endl;
      }
   }
}

//Retrieves values of a vector of variant and outputs them to a xml file
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

   // Retrives value from variant
   for (const multipleTypes &element : singleBuffer_) {
      if (basicType == typeid(uint64_t).name()) {
         os << get<uint64_t>(element) << " ";
      } else if (basicType == typeid(bool).name()) {
         os << get<bool>(element) << " ";
      } else if (basicType == typeid(int).name()) {
         os << get<int>(element) << " ";
      } else if (basicType == typeid(BGFLOAT).name()) {
         os << get<BGFLOAT>(element) << " ";
      } else if (basicType == typeid(vertexType).name()) {
         os << static_cast<int>(get<vertexType>(element)) << " ";
      } else if (basicType == typeid(double).name()) {
         os << get<double>(element) << " ";
      } else if (basicType == typeid(unsigned char).name()) {
         os << get<unsigned char>(element) << " ";
      } else {
         perror("Error recording Recordable object");
         exit(EXIT_FAILURE);
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

/// Receives a recorded variable entity from the variable owner class
/**
* @brief Registers a single instance of a class derived from RecordableBase.
* @param varName Name of the recorded variable.
* @param recordVar Reference to the recorded variable.
* @param variableType Type of the recorded variable.
*/
void XmlRecorder::registerVariable(const string &varName, RecordableBase &recordVar,
                                   UpdatedType variableType)
{
   variableTable_.push_back(singleVariableInfo(varName, recordVar, variableType));
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
void XmlRecorder::registerVariable(const string &varName, vector<RecordableBase *> &recordVars,
                                   UpdatedType variableType)
{
   for (int i = 0; i < recordVars.size(); i++) {
      string variableID = varName + to_string(i);
      RecordableBase &address = *recordVars[i];
      // add a new variable into the table
      variableTable_.push_back(singleVariableInfo(variableID, address, variableType));
   }
}
