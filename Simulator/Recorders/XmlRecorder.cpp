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
/// @param[in] vertices will be removed eventually
void XmlRecorder::compileHistories(AllVertices &vertices)
{
   for (int rowIndex = 0; rowIndex < variableTable_.size(); rowIndex++) {
      if (variableTable_[rowIndex].variableLocation_->getNumEventsInEpoch() > 0) {
         for (int columnIndex = 0;
              columnIndex < variableTable_[rowIndex].variableLocation_->getNumEventsInEpoch();
              columnIndex++) {
            // cout << (*(variableTable_[i].variableLocation_))[j] << " ";
            variableTable_[rowIndex].variableHistory_.push_back(
               (*(variableTable_[rowIndex].variableLocation_)).getElement(columnIndex)
                  );
         }
      }
         // cout << endl;
         variableTable_[rowIndex].variableLocation_->startNewEpoch();
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

/// Writes simulation results to an output destination.
/// @param  vertices will be removed eventually.
void XmlRecorder::saveSimData(const AllVertices &vertices)
{
   // Write XML header information:
   resultOut_ << "<?xml version=\"1.0\" standalone=\"no\"?>\n";
   //iterate the variable list row by row then output the cumulative value to a xml file
   for (int rowIndex = 0; rowIndex < variableTable_.size(); rowIndex++) {
      if (variableTable_[rowIndex].variableHistory_.size() > 0) {
         resultOut_ << toXML(variableTable_[rowIndex].variableName_,
            variableTable_[rowIndex].variableHistory_)
                    << endl;
      }
   }
}

string XmlRecorder::toXML(string name, vector<multipleTypes> singleBuffer_) const
{
    stringstream os;

    os << "<Matrix ";
    if (!name.empty())
        os << "name=\"" << name << "\" ";
    os << "type=\"complete\" rows=\"" << 1 << "\" columns=\"" << singleBuffer_.size()
       << "\" multiplier=\"1.0\">" << endl;
    
    os << "   ";
   //  for (const auto& value : singleBuffer_) {
   //      std::visit([&os](const auto& v) {
   //          os << v << " ";
   //      }, value);
   //  }

   // for (const auto& element : singleBuffer_) {
   //  std::visit(XmlRecorder::VariantVisitor{}, os, element);
   // }

   for (const multipleTypes& element : singleBuffer_) {
      if(holds_alternative<uint64_t>(element)){
         os << get<uint64_t>(element);
      }else if(holds_alternative<double>(element)){
         os << get<double>(element);
      }else if(holds_alternative<string>(element)){
         os << get<string>(element);
      }
 
   }
   // constexpr size_t idx = singleBuffer_[0].
   //    for (const multipleTypes& element : singleBuffer_) {

   //  }

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

/// register a single EventBuffer.
/// Obtain the updating value while the simulator runs by storing the address of registered variable
/// Store a single neuron with the neuron number and its corresponding events
void XmlRecorder::registerVariable(string name, RecordableBase &recordVar)
{
   // add a new variable into the table
   variableTable_.push_back(singleVariableInfo(name, recordVar));
}

/// register a vector of EventBuffers.
/// Obtain the updating value while the simulator runs by storing the address of registered variable
/// Store all neuron with the neuron number and its corresponding events
void XmlRecorder::registerVariable(string varName, vector<RecordableBase> &recordVars)
{
   for (int i = 0; i < recordVars.size(); i++) {
      string variableID = varName + to_string(i);
      // add a new variable into the table
      variableTable_.push_back(singleVariableInfo(variableID, recordVars[i]));
   }
}
