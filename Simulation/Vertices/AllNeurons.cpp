#include "AllNeurons.h"
#include "Core/Simulator.h"
#include "OperationManager.h"

// Default constructor
AllNeurons::AllNeurons() :
      size_(0) {
   summationMap_ = NULL;

   // Register loadParameters function as a loadParameters operation in the Operation Manager
   auto loadParametersFunc = std::bind(&IAllNeurons::loadParameters, this);
   OperationManager::getInstance().registerOperation(Operations::op::loadParameters, loadParametersFunc);

   // Register printParameters function as a printParameters operation in the OperationManager
   function<void()> printParametersFunc = bind(&IAllNeurons::printParameters, this);
   OperationManager::getInstance().registerOperation(Operations::printParameters, printParametersFunc);
}

AllNeurons::~AllNeurons() {
   freeResources();
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 */
void AllNeurons::setupNeurons() {
   size_ = Simulator::getInstance().getTotalNeurons();
   summationMap_ = new BGFLOAT[size_];

   for (int i = 0; i < size_; ++i) {
      summationMap_[i] = 0;
   }

   Simulator::getInstance().setPSummationMap(summationMap_);
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllNeurons::cleanupNeurons() {
   freeResources();
}

/*
 *  Deallocate all resources
 */
void AllNeurons::freeResources() {
   if (size_ != 0) {
      delete[] summationMap_;
   }

   summationMap_ = NULL;

   size_ = 0;
}

void AllNeurons::printParameters() const {
   cout << "VERTICE PARAMETERS" << endl;
}
