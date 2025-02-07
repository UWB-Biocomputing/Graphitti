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
}

///  Setup the internal structure of the class (allocate memories).
void AllVertices::setupVertices()
{
   size_ = Simulator::getInstance().getTotalVertices();
#if defined(USE_GPU)
   // We don't allocate memory for summationPoints_ in CPU when building the GPU
   // implementation. This is to avoid misusing it in GPU code.
   // summationPoints_ = nullptr;

#else
   summationPoints_.assign(size_, 0);

#endif
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
