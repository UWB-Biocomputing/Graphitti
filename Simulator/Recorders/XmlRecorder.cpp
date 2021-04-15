/**
 * @file XmlRecorder.cpp
 *
 * @ingroup Simulator/Recorders
 * 
 * @brief An implementation for recording spikes history on xml file
 */

#include <functional>

#include "XmlRecorder.h"
#include "AllIFNeurons.h"      // TODO: remove LIF model specific code
#include "ConnGrowth.h"
#include "OperationManager.h"
#include "ParameterManager.h"

/// constructor
XmlRecorder::XmlRecorder() {
   resultFileName_ = Simulator::getInstance().getResultFileName();

   function<void()> printParametersFunc = std::bind(&XmlRecorder::printParameters, this);
   OperationManager::getInstance().registerOperation(Operations::printParameters, printParametersFunc);

   fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));
}

/// destructor
XmlRecorder::~XmlRecorder() {
}

/// Initialize data
/// Create a new xml file.
///
/// @param[in] stateOutputFileName	File name to save histories
void XmlRecorder::init() {
   stateOut_.open(resultFileName_.c_str());
}

/// Init radii and rates history matrices with default values
void XmlRecorder::initDefaultValues() {
}

/// Init radii and rates history matrices with current radii and rates
void XmlRecorder::initValues() {
}

/// Get the current radii and rates values
void XmlRecorder::getValues() {
}

/// Terminate process
void XmlRecorder::term() {
   stateOut_.close();
}
