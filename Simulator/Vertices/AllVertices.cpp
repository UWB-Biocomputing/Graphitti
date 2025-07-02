/**
 * @file AllVertices.cpp
 * 
 * @ingroup Simulator/Vertices
 *
 * @brief A container of the base class of all vertex data
 */

#include "AllVertices.h"
#include "OperationManager.h"

// Default constructor
AllVertices::AllVertices() : size_(0)
{
   // Register loadParameters function as a loadParameters operation in the Operation Manager
   function<void()> loadParametersFunc = std::bind(&AllVertices::loadParameters, this);
   OperationManager::getInstance().registerOperation(Operations::loadParameters,
                                                     loadParametersFunc);

   // Register printParameters function as a printParameters operation in the OperationManager
   function<void()> printParametersFunc = bind(&AllVertices::printParameters, this);
   OperationManager::getInstance().registerOperation(Operations::printParameters,
                                                     printParametersFunc);

   // Get a copy of the file and vertex logger to use log4cplus macros to print to debug files
   fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));
   vertexLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("vertex"));
   vertexLogger_.setLogLevel(log4cplus::DEBUG_LOG_LEVEL);
}

///  Setup the internal structure of the class (allocate memories).
void AllVertices::setupVertices()
{
   size_ = Simulator::getInstance().getTotalVertices();
}

///  Prints out all parameters of the vertices to logging file.
///  Registered to OperationManager as Operation::printParameters
void AllVertices::printParameters() const
{
   LOG4CPLUS_DEBUG(fileLogger_, "\nVERTICES PARAMETERS");
}

/// Loads all inputs scheduled to occur in the upcoming epoch.
/// These are inputs occurring in between curStep (inclusive) and
/// endStep (exclusive)
void AllVertices::loadEpochInputs(uint64_t currentStep, uint64_t endStep)
{
   // This is an empty implementation so that Neural Network simulation works
   // normally
}


#ifdef USE_GPU
/// Set the CUDA stream to be used by GPU vertices kernels in derived classes.
///
/// This assigns a CUDA stream to the base class, allowing subclasses
/// to launch kernels on the correct stream. The stream is typically
/// created by GPUModel and passed down during simulation setup.
///
/// @param simulationStream A valid CUDA stream (`cudaStream_t`) managed by the caller.
void AllVertices::SetStream(cudaStream_t simulationStream)
{
   simulationStream_ = simulationStream;
}
#endif