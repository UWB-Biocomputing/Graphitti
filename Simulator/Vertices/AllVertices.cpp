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
   OperationManager::getInstance().registerOperation(Operations::op::loadParameters,
                                                     loadParametersFunc);

   // Register printParameters function as a printParameters operation in the OperationManager
   function<void()> printParametersFunc = bind(&AllVertices::printParameters, this);
   OperationManager::getInstance().registerOperation(Operations::printParameters,
                                                     printParametersFunc);

   // Register registerHistoryVariables function as a registerHistoryVariables operation in the OperationManager
   function<void()> registerHistory = bind(&AllVertices::registerHistoryVariables, this);
   OperationManager::getInstance().registerOperation(Operations::registerHistoryVariables,
                                                      registerHistory);

#if defined(USE_GPU)
   // Register allocNeuronDeviceStruct function as a allocateGPU operation in the OperationManager
   function<void()> allocateGPU = bind(&AllVertices::allocNeuronDeviceStruct, this);
   OperationManager::getInstance().registerOperation(Operations::allocateGPU, allocateGPU);

   // Register AllVertices::copyToDevice function as a copyToGPU operation in the OperationManager
   function<void()> copyCPUtoGPU = bind(&AllVertices::copyToDevice, this);
   OperationManager::getInstance().registerOperation(Operations::copyToGPU, copyCPUtoGPU);

   // Register copyFromGPU operation for transferring edge data from device to host
   function<void()> copyFromGPU = bind(&AllVertices::copyFromDevice, this);
   OperationManager::getInstance().registerOperation(Operations::copyFromGPU, copyFromGPU);

   // Register deleteNeuronDeviceStruct function as a deallocateGPUMemory operation in the OperationManager
   function<void()> deallocateGPUMemory = bind(&AllVertices::deleteNeuronDeviceStruct, this);
   OperationManager::getInstance().registerOperation(Operations::deallocateGPUMemory,
                                                     deallocateGPUMemory);
#endif

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
