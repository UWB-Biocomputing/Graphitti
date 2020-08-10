#include "AllNeurons.h"
#include "Core/Simulator.h"

// Default constructor
AllNeurons::AllNeurons() :
      size_(0) {
   summationMap_ = NULL;
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
