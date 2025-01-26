/**
 * @file CPUModel.cpp
 * 
 * @ingroup Simulator/Core
 *
 * @brief Implementation of Model for graph-based networks.
 */

#include "CPUModel.h"
#include "AllDSSynapses.h"
#include "Simulator.h"

#if !defined(USE_GPU)

/// Performs any finalization tasks on network following a simulation.
void CPUModel::finish()
{
   // No GPU code to deallocate, and CPU side deallocation is handled by destructors.
}

/// Advance everything in the model one time step.
void CPUModel::advance()
{
   // ToDo: look at pointer v no pointer in params - to change
   // dereferencing the ptr, lose late binding -- look into changing!
   AllVertices &vertices = layout_->getVertices();
   AllEdges &edges = connections_->getEdges();
   EdgeIndexMap &edgeIndexMap = connections_->getEdgeIndexMap();

   vertices.advanceVertices(edges, edgeIndexMap);
   edges.advanceEdges(vertices, edgeIndexMap);
   vertices.integrateVertexInputs(edges, edgeIndexMap);
}

/// Update the connection of all the Neurons and Synapses of the simulation.
void CPUModel::updateConnections()
{
   // Update Connections data
   if (connections_->updateConnections(layout_->getVertices())) {
      connections_->updateSynapsesWeights();
      // create synapse inverse map
      connections_->createEdgeIndexMap();
   }
}

/// Copy GPU Synapse data to CPU. (Inheritance, no implem)
void CPUModel::copyGPUtoCPU()
{
   LOG4CPLUS_WARN(fileLogger_, "ERROR: CPUModel::copyGPUtoCPU() was called." << endl);
   exit(EXIT_FAILURE);
}

/// Copy CPU Synapse data to GPU. (Inheritance, no implem, GPUModel has implem)
void CPUModel::copyCPUtoGPU()
{
   LOG4CPLUS_WARN(fileLogger_, "ERROR: CPUModel::copyCPUtoGPU() was called." << endl);
   exit(EXIT_FAILURE);
}
#endif   // define(USE_GPU)