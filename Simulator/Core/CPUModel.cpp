/**
 * @file CPUModel.cpp
 * 
 * @ingroup Simulator/Core
 *
 * @brief Implementation of Model for graph-based networks.
 */

#include "CPUModel.h"
#include "Simulator.h"
#include "AllDSSynapses.h"

#if !defined(USE_GPU)

/// Constructor
CPUModel::CPUModel() : Model() {
}

/// Destructor
CPUModel::~CPUModel() {
   // Let Model base class handle de-allocation
}

/// Performs any finalization tasks on network following a simulation.
void CPUModel::finish() {
   // No GPU code to deallocate, and CPU side deallocation is handled by destructors.
}

/// Advance everything in the model one time step.
void CPUModel::advance() {
   // ToDo: look at pointer v no pointer in params - to change
   // dereferencing the ptr, lose late binding -- look into changing!
   layout_->getVertices()->advanceVertices(*(connections_->getEdges().get()), connections_->getEdgeIndexMap().get());
   connections_->getEdges()->advanceEdges(layout_->getVertices().get(), connections_->getEdgeIndexMap().get());
}

/// Update the connection of all the Neurons and Synapses of the simulation.
void CPUModel::updateConnections() {
   // Update Connections data
   if (connections_->updateConnections(*layout_->getVertices(), layout_.get())) {
      connections_->updateSynapsesWeights(
            Simulator::getInstance().getTotalVertices(),
            *layout_->getVertices(),
            *connections_->getEdges(),
            layout_.get());
      // create synapse inverse map
      connections_->createEdgeIndexMap();
   }
}

/// Copy GPU Synapse data to CPU. (Inheritance, no implem)
void CPUModel::copyGPUtoCPU() {
   LOG4CPLUS_WARN(fileLogger_, "ERROR: CPUModel::copyGPUtoCPU() was called." << endl);
   exit(EXIT_FAILURE);
}

/// Copy CPU Synapse data to GPU. (Inheritance, no implem, GPUModel has implem)
void CPUModel::copyCPUtoGPU() {
   LOG4CPLUS_WARN(fileLogger_, "ERROR: CPUModel::copyCPUtoGPU() was called." << endl);
   exit(EXIT_FAILURE);
}
# endif // define(USE_GPU)