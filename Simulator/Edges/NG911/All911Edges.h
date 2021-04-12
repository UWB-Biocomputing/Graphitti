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

   ///  Creates an instance of the class.
   ///
   ///  @return Reference to the instance of the class.
   static IAllEdges *Create() { return new All911Edges(); }

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
                              edgeType type);

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

#endif // defined(USE_GPU)

