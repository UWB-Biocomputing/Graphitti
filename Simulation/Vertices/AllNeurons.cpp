/**
 * @file AllNeurons.cpp
 * 
 * @ingroup Simulation/Vertices
 *
 * @brief
 */

#include "AllNeurons.h"
#include "Core/Simulator.h"
#include "OperationManager.h"

// Default constructor
AllNeurons::AllNeurons() : size_(0) {
   summationMap_ = NULL;

   // Register loadParameters function as a loadParameters operation in the Operation Manager
   auto loadParametersFunc = std::bind(&IAllNeurons::loadParameters, this);
   OperationManager::getInstance().registerOperation(Operations::op::loadParameters, loadParametersFunc);

   // Register printParameters function as a printParameters operation in the OperationManager
   function<void()> printParametersFunc = bind(&IAllNeurons::printParameters, this);
   OperationManager::getInstance().registerOperation(Operations::printParameters, printParametersFunc);

   // Get a copy of the file and neuron logger to use log4cplus macros to print to debug files
   fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));
   neuronLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("neuron"));
}

AllNeurons::~AllNeurons() {
   if (size_ != 0) {
      delete[] summationMap_;
   }

   summationMap_ = NULL;

   size_ = 0;
}

///  Setup the internal structure of the class (allocate memories).
void AllNeurons::setupNeurons() {
   size_ = Simulator::getInstance().getTotalNeurons();
   summationMap_ = new BGFLOAT[size_];

   for (int i = 0; i < size_; ++i) {
      summationMap_[i] = 0;
   }

   Simulator::getInstance().setPSummationMap(summationMap_);
}

///  Prints out all parameters of the neurons to logging file.
///  Registered to OperationManager as Operation::printParameters
void AllNeurons::printParameters() const {
   LOG4CPLUS_DEBUG(fileLogger_, "\nVERTICES PARAMETERS");
}
