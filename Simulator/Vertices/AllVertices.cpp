/**
 * @file AllVertices.cpp
 * 
 * @ingroup Simulator/Vertices
 *
 * @brief A container of the base class of all vertex data
 */

#include "AllVertices.h"
#include "OperationManager.h"

// Utility function to convert a vertexType into a string.
// MODEL INDEPENDENT FUNCTION NMV-BEGIN {
string vertexTypeToString(vertexType t)
{
   switch (t) {
      case vertexType::INH:
         return "INH";
      case vertexType::EXC:
         return "EXC";
      default:
         cerr << "ERROR->vertexTypeToString() failed, unknown type: " << t << endl;
         assert(false);
         return nullptr;   // Must return a value -- this will probably cascade to another failure
   }
}
// } NMV-END

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
   
   function<void(uint64_t,uint64_t)> loadEpochInputsFunc = std::bind(&AllVertices::loadEpochInputs, 
                                                                     this, 
                                                                     std::placeholders::_1, 
                                                                     std::placeholders::_2);
   OperationManager::getInstance().registerOperation(Operations::loadEpochInputs,
                                                     loadEpochInputsFunc);


   // Register registerHistoryVariables function as a registerHistoryVariables operation in the OperationManager
   function<void()> registerHistoryVarsFunc = bind(&AllVertices::registerHistoryVariables, this);
   OperationManager::getInstance().registerOperation(Operations::registerHistoryVariables,
                                                     registerHistoryVarsFunc);

#if defined(USE_GPU)
   // Register allocNeuronDeviceStruct function as a allocateGPU operation in the OperationManager
   function<void()> allocateGPU = bind(&AllVertices::allocVerticesDeviceStruct, this);
   OperationManager::getInstance().registerOperation(Operations::allocateGPU, allocateGPU);

   // Register AllVertices::copyToDevice function as a copyToGPU operation in the OperationManager
   function<void()> copyCPUtoGPU = bind(&AllVertices::copyToDevice, this);
   OperationManager::getInstance().registerOperation(Operations::copyToGPU, copyCPUtoGPU);

   // Register copyFromGPU operation for transferring edge data from device to host
   function<void()> copyFromGPU = bind(&AllVertices::copyFromDevice, this);
   OperationManager::getInstance().registerOperation(Operations::copyFromGPU, copyFromGPU);

   // Register deleteNeuronDeviceStruct function as a deallocateGPUMemory operation in the OperationManager
   function<void()> deallocateGPUMemory = bind(&AllVertices::deleteVerticesDeviceStruct, this);
   OperationManager::getInstance().registerOperation(Operations::deallocateGPUMemory,
                                                     deallocateGPUMemory);
#endif

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
   loadEpochInputsToVertices(currentStep, endStep);
   #if defined(USE_GPU)
   copyEpochInputsToDevice();
   #endif
}

void AllVertices::loadEpochInputsToVertices(uint64_t currentStep, uint64_t endStep)
{
   // This is an empty implementation so that Neural Network simulation works
   // normally
   LOG4CPLUS_DEBUG(vertexLogger_, "Calling AllVertices::loadEpochInputsToVertices");
}

#if defined(USE_GPU)
void AllVertices::copyEpochInputsToDevice()
{
   // This is an empty implementation so that Neural Network simulation works
   // normally
   LOG4CPLUS_DEBUG(vertexLogger_, "Calling AllVertices::copyEpochInputsToDevice");
}

int AllVertices::getNumberOfVerticesNeedingDeviceNoise() const
{
   return Simulator::getInstance().getTotalVertices();
}
#endif