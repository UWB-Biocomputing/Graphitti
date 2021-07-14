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
AllVertices::AllVertices() : size_(0) {
   summationMap_ = nullptr;

   // Register loadParameters function as a loadParameters operation in the Operation Manager
   function<void()> loadParametersFunc = std::bind(&AllVertices::loadParameters, this);
   OperationManager::getInstance().registerOperation(Operations::op::loadParameters, loadParametersFunc);

   // Register printParameters function as a printParameters operation in the OperationManager
   function<void()> printParametersFunc = bind(&AllVertices::printParameters, this);
   OperationManager::getInstance().registerOperation(Operations::printParameters, printParametersFunc);

   // Get a copy of the file and vertex logger to use log4cplus macros to print to debug files
   fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));
   vertexLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("vertex"));
}

AllVertices::~AllVertices() {
   if (size_ != 0) {
      delete[] summationMap_;
   }

   summationMap_ = nullptr;

   size_ = 0;
}

///  Setup the internal structure of the class (allocate memories).
void AllVertices::setupVertices() {
   size_ = Simulator::getInstance().getTotalVertices();
   summationMap_ = new BGFLOAT[size_];

   for (int i = 0; i < size_; ++i) {
      summationMap_[i] = 0;
   }

   Simulator::getInstance().setPSummationMap(summationMap_);
}

///  Prints out all parameters of the vertices to logging file.
///  Registered to OperationManager as Operation::printParameters
void AllVertices::printParameters() const {
   LOG4CPLUS_DEBUG(fileLogger_, "\nVERTICES PARAMETERS");
}
