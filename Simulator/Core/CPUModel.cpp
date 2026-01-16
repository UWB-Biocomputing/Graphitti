/**
 * @file CPUModel.cpp
 * 
 * @ingroup Simulator/Core
 *
 * @brief Implementation of Model for graph-based networks.
 */

#include "CPUModel.h"
#include "Simulator.h"
#include "Timer.h"

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

   log4cplus::Logger consoleLogger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("console"));
   // Elapsed time in us
   double elapsedTime = 0.0;
   timer.start();
   vertices.advanceVertices(edges, edgeIndexMap);
   elapsedTime = timer.lap();
   LOG4CPLUS_TRACE(consoleLogger, "advanceVertices time: " << elapsedTime);
   timer.start();
   edges.advanceEdges(vertices, edgeIndexMap);
   elapsedTime = timer.lap();
   LOG4CPLUS_TRACE(consoleLogger, "advanceEdges time: " << elapsedTime);
   timer.start();
   vertices.integrateVertexInputs(edges, edgeIndexMap);
   elapsedTime = timer.lap();
   LOG4CPLUS_TRACE(consoleLogger, "integrateVertexInputs time: " << elapsedTime);
}

/// Update the connection of all the vertices and edges of the simulation.
void CPUModel::updateConnections()
{
   // Update Connections data
   if (connections_->updateConnections()) {
      connections_->updateEdgesWeights();
      // create edge inverse map
      connections_->createEdgeIndexMap();
   }
}

/// Copy GPU edge data to CPU. (Inheritance, no implem)
void CPUModel::copyGPUtoCPU()
{
   LOG4CPLUS_WARN(fileLogger_, "ERROR: CPUModel::copyGPUtoCPU() was called." << endl);
   exit(EXIT_FAILURE);
}

/// Copy CPU edge data to GPU. (Inheritance, no implem, GPUModel has implem)
void CPUModel::copyCPUtoGPU()
{
   LOG4CPLUS_WARN(fileLogger_, "ERROR: CPUModel::copyCPUtoGPU() was called." << endl);
   exit(EXIT_FAILURE);
}
#endif   // define(USE_GPU)