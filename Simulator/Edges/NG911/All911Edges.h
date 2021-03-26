/**
 *  @file All911Edges.h
 * 
 *  @ingroup Simulator/Edges/NG911
 *
 *  @brief A container of all 911 edge data
 *
 *  The container holds edge parameters of all edges. 
 *  Each kind of edge parameter is stored in a 2D array. Each item in the first 
 *  dimention of the array corresponds with each vertex, and each item in the second
 *  dimension of the array corresponds with an edge parameter of each edge of the vertex. 
 *  Bacause each vertex owns different number of edges, the number of edges 
 *  for each vertex is stored in a 1D array, edge_counts.
 *
 *  For CUDA implementation, we used another structure, AllEdgesDevice, where edge
 *  parameters are stored in 1D arrays instead of 2D arrays, so that device functions
 *  can access these data less latency. When copying a edge parameter, P[i][j],
 *  from host to device, it is stored in P[i * max_edges_per_vertex + j] in 
 *  AllEdgesDevice structure.
 *
 */
#pragma once

#include "AllEdges.h"

struct All911EdgeDeviceProperties;

class All911Edges : public AllEdges {
public:
   All911Edges();

   All911Edges(const int numVertices, const int maxEdges);

   virtual ~All911Edges();

   static IAllEdges *Create() {
      return new All911Edges();
   }

   ///  Setup the internal structure of the class (allocate memories and initialize them).
   virtual void setupEdges();

   ///  Create a Edge and connect it to the model.
   ///
   ///  @param  iEdg        Index of the edge to set.
   ///  @param  srcVertex   Coordinates of the source Neuron.
   ///  @param  destVertex  Coordinates of the destination Neuron.
   ///  @param  sumPoint    Summation point address.
   ///  @param  deltaT      Inner simulation step duration.
   ///  @param  type        Type of the Edge to create.
   virtual void createEdge(const BGSIZE iEdg, int srcVertex, int destVertex, BGFLOAT *sumPoint, const BGFLOAT deltaT,
                              synapseType type);

   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const;

protected:
   ///  Setup the internal structure of the class (allocate memories and initialize them).
   ///
   ///  @param  numVertices   Total number of vertices in the network.
   ///  @param  maxEdges  Maximum number of edges per vertex.
   virtual void setupEdges(const int numVertices, const int maxEdges);

#if defined(USE_GPU)
   public:
       ///  Allocate GPU memories to store all edges' states,
       ///  and copy them from host to GPU memory.
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       virtual void allocEdgeDeviceStruct( void** allEdgesDevice );

       ///  Allocate GPU memories to store all edges' states,
       ///  and copy them from host to GPU memory.
       ///
       ///  @param  allEdgesDevice     GPU address of the allEdges struct on device memory.
       ///  @param  numVertices            Number of vertices.
       ///  @param  maxEdgesPerVertex  Maximum number of edges per vertex.
       virtual void allocEdgeDeviceStruct( void** allEdgesDevice, int numVertices, int maxEdgesPerVertex );

       ///  Delete GPU memories.
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       virtual void deleteEdgeDeviceStruct( void* allEdgesDevice );

       ///  Copy all edges' data from host to device.
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       virtual void copyEdgeHostToDevice( void* allEdgesDevice );

       ///  Copy all edges' data from host to device.
       ///
       ///  @param  allEdgesDevice     GPU address of the allEdges struct on device memory.
       ///  @param  numVertices            Number of vertices.
       ///  @param  maxEdgesPerVertex  Maximum number of edges per vertex.
       virtual void copyEdgeHostToDevice( void* allEdgesDevice, int numVertices, int maxEdgesPerVertex );
       
       /// Copy all edges' data from device to host.
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       virtual void copyEdgeDeviceToHost( void* allEdgesDevice );

       ///  Get edge_counts in AllEdges struct on device memory.
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       virtual void copyDeviceEdgeCountsToHost( void* allEdgesDevice );

       ///  Get summationCoord and in_use in AllEdges struct on device memory.
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       virtual void copyDeviceEdgeSumIdxToHost( void* allEdgesDevice );

       ///  Advance all the Edges in the simulation.
       ///  Update the state of all edges for a time step.
       ///
       ///  @param  allEdgesDevice      GPU address of the allEdges struct on device memory.
       ///  @param  allVerticesDevice       GPU address of the allVertices struct on device memory.
       ///  @param  edgeIndexMapDevice  GPU address of the EdgeIndexMap on device memory.
       virtual void advanceEdges( void* allEdgesDevice, void* allVerticesDevice, void* edgeIndexMapDevice );

       ///  Set some parameters used for advanceSynapsesDevice.
       ///  Currently we set a member variable: m_fpChangePSR_h.
       virtual void setAdvanceEdgesDeviceParams( );

       ///  Set edge class ID defined by enumClassSynapses for the caller's Edge class.
       ///  The class ID will be set to classSynapses_d in device memory,
       ///  and the classSynapses_d will be referred to call a device function for the
       ///  particular edge class.
       ///  Because we cannot use virtual function (Polymorphism) in device functions,
       ///  we use this scheme.
       ///  Note: we used to use a function pointer; however, it caused the growth_cuda crash
       ///  (see issue#137).
       virtual void setEdgeClassID( );

       ///  Prints GPU SynapsesProps data.
       ///
       ///  @param  allEdgesDeviceProps   GPU address of the corresponding SynapsesDeviceProperties struct on device memory.
       virtual void printGPUEdgesProps( void* allEdgesDeviceProps ) const;

   protected:
       ///  Allocate GPU memories to store all edges' states,
       ///  and copy them from host to GPU memory.
       ///  (Helper function of allocEdgeDeviceStruct)
       ///
       ///  @param  allEdgesDevice     GPU address of the allEdges struct on device memory.
       ///  @param  numVertices            Number of vertices.
       ///  @param  maxEdgesPerVertex  Maximum number of edges per vertex.
       void allocDeviceStruct( All911EdgeDeviceProperties &allEdgesDevice, int numVertices, int maxEdgesPerVertex );

       ///  Delete GPU memories.
       ///  (Helper function of deleteEdgeDeviceStruct)
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       void deleteDeviceStruct( All911EdgeDeviceProperties& allEdgesDevice );

       ///  Copy all edges' data from host to device.
       ///  (Helper function of copyEdgeHostToDevice)
       ///
       ///  @param  allEdgesDevice     GPU address of the allEdges struct on device memory.
       ///  @param  numVertices            Number of vertices.
       ///  @param  maxEdgesPerVertex  Maximum number of edges per vertex.
       void copyHostToDevice( void* allEdgesDevice, All911EdgeDeviceProperties& allEdgesDeviceProps, int numVertices, int maxEdgesPerVertex );

       ///  Copy all edges' data from device to host.
       ///  (Helper function of copyEdgeDeviceToHost)
       ///
       ///  @param  allEdgesDevice     GPU address of the allEdges struct on device memory.
       ///  @param  numVertices            Number of vertices.
       ///  @param  maxEdgesPerVertex  Maximum number of edges per vertex.
       void copyDeviceToHost( All911EdgeDeviceProperties& allEdgesDevice);

#else  // !defined(USE_GPU)
public:
   ///  Advance one specific Edge.
   ///
   ///  @param  iEdg      Index of the Edge to connect to.
   ///  @param  vertices   The Neuron list to search from.
   virtual void advanceEdge(const BGSIZE iEdg, IAllVertices *vertices);

#endif
};

#if defined(USE_GPU)

struct All911EdgeDeviceProperties : public AllEdgesDeviceProperties{};

#endif // defined(USE_GPU)

