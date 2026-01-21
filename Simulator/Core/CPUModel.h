/**
 * @file CPUModel.h
 * 
 * @ingroup Simulator/Core
 *
 * @brief Implementation of Model for execution on CPU (single core).
 * 
 * The Model class maintains and manages classes of objects that make up
 * essential components of the graph network.
 *    -# AllVertices: A class to define a list of particular type of vertices.
 *    -# AllEdges: A class to define a list of particular type of edges.
 *    -# Connections: A class to define connections of the graph network.
 *    -# Layout: A class to define vertices' layout information in the network.
 *
 * Edges in the edge map are located at the coordinates of the vertex
 * from which they receive output.
 *
 * The model runs on a single thread.
 *
 */

#pragma once

#include "Connections/Connections.h"
#include "Edges/AllEdges.h"
#include "Layouts/Layout.h"
#include "Vertices/AllVertices.h"

class CPUModel : public Model {
public:
   /// Constructor
   CPUModel() = default;

   /// Destructor
   virtual ~CPUModel() = default;

   /// Performs any finalization tasks on network following a simulation.
   virtual void finish() override;

   /// Advances network state one simulation step.
   virtual void advance() override;

   /// Modifies connections between vertices based on current state of the network and behavior
   /// over the past epoch. Should be called once every epoch.
   virtual void updateConnections() override;

   /// Copy GPU edge data to CPU.
   virtual void copyGPUtoCPU() override;

   /// Copy CPU edge data to GPU.
   virtual void copyCPUtoGPU() override;
};
