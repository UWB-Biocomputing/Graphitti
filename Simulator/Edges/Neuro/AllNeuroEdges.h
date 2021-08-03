/**
 *  @file AllNeuroEdges.h
 *
 *  @ingroup Simulator/Neuro/Edges
 * 
 *  @brief A container of all edge data
 *
 *  The container holds edge parameters of all edges. 
 *  Each kind of edge parameter is stored in a 2D array. Each item in the first 
 *  dimention of the array corresponds with each vertex, and each item in the second
 *  dimension of the array corresponds with a edge parameter of each edge of the vertex. 
 *  Bacause each vertex owns different number of edges, the number of edges 
 *  for each vertex is stored in a 1D array, edge_counts.
 *
 *  For CUDA implementation, we used another structure, AllEdgesDevice, where edge
 *  parameters are stored in 1D arrays instead of 2D arrays, so that device functions
 *  can access these data less latency. When copying a edge parameter, P[i][j],
 *  from host to device, it is stored in P[i * max_edges_per_vertex + j] in 
 *  AllEdgesDevice structure.
 *
 *  The latest implementation uses the identical data struture between host and CUDA;
 *  that is, edge parameters are stored in a 1D array, so we don't need conversion 
 *  when copying data between host and device memory.
 */

#pragma once

#include "AllEdges.h"

#ifdef _WIN32
typedef unsigned _int8 uint8_t;
#endif

class AllVertices;

// typedef void (*fpCreateSynapse_t)(void*, const int, const int, int, int, BGFLOAT*, const BGFLOAT, edgeType);

// enumerate all non-abstract edge classes.
enum enumClassSynapses {classAllSpikingSynapses, classAllDSSynapses, classAllSTDPSynapses, classAllDynamicSTDPSynapses, undefClassSynapses};

class AllNeuroEdges : public AllEdges {
public:
   AllNeuroEdges();

   virtual ~AllNeuroEdges();

   ///  Setup the internal structure of the class (allocate memories and initialize them).
   virtual void setupEdges() override;

   ///  Reset time varying state vars and recompute decay.
   ///
   ///  @param  iEdg     Index of the edge to set.
   ///  @param  deltaT   Inner simulation step duration
   virtual void resetEdge(const BGSIZE iEdg, const BGFLOAT deltaT);

   // ///  Create a Synapse and connect it to the model.
   // ///
   // ///  @param  iEdg        Index of the edge to set.
   // ///  @param  source      Coordinates of the source Neuron.
   // ///  @param  dest        Coordinates of the destination Neuron.
   // ///  @param  sumPoint    Summation point address.
   // ///  @param  deltaT      Inner simulation step duration.
   // ///  @param  type        Type of the Synapse to create.
   // virtual void createEdge(const BGSIZE iEdg, int srcVertex, int destVertex, BGFLOAT *sumPoint, const BGFLOAT deltaT,
   //                            edgeType type) override;

   ///  Get the sign of the edgeType.
   ///
   ///  @param    type    edgeType I to I, I to E, E to I, or E to E
   ///  @return   1 or -1, or 0 if error
   int edgSign(const edgeType type);

   ///  Prints SynapsesProps data to console.
   virtual void printSynapsesProps() const;

protected:
   ///  Setup the internal structure of the class (allocate memories and initialize them).
   ///
   ///  @param  numVertices   Total number of vertices in the network.
   ///  @param  maxEdges  Maximum number of edges per vertex.
   virtual void setupEdges(const int numVertices, const int maxEdges) override;

   ///  Sets the data for Synapse to input's data.
   ///
   ///  @param  input  istream to read from.
   ///  @param  iEdg   Index of the edge to set.
   virtual void readEdge(istream &input, const BGSIZE iEdg) override;

   ///  Write the edge data to the stream.
   ///
   ///  @param  output  stream to print out to.
   ///  @param  iEdg    Index of the edge to print out.
   virtual void writeEdge(ostream &output, const BGSIZE iEdg) const override;

public:
   /// The factor to adjust overlapping area to edge weight.
   static constexpr BGFLOAT SYNAPSE_STRENGTH_ADJUSTMENT = 1.0e-8;
      ///  The post-synaptic response is the result of whatever computation
   ///  is going on in the edge.
   BGFLOAT *psr_;
};

#if defined(USE_GPU)
struct AllEdgesDeviceProperties
{
        ///  The location of the edge.
        int *sourceVertexIndex_;

        ///  The coordinates of the summation point.
        int *destVertexIndex_;

        ///   The weight (scaling factor, strength, maximal amplitude) of the edge.
         BGFLOAT *W_;

        ///  Synapse type
        edgeType *type_;

        ///  The post-synaptic response is the result of whatever computation
        ///  is going on in the edge.
        BGFLOAT *psr_;

        ///  The boolean value indicating the entry in the array is in use.
        bool *inUse_;

        ///  The number of edges for each vertex.
        ///  Note: Likely under a different name in GpuSim_struct, see edge_count. -Aaron
        BGSIZE *edgeCounts_;

        ///  The total number of active edges.
        BGSIZE totalEdgeCount_;

        ///  The maximum number of edges for each vertex.
        BGSIZE maxEdgesPerVertex_;

        ///  The number of vertices
        ///  Aaron: Is this even supposed to be here?!
        ///  Usage: Used by destructor
        int countVertices_;
};
#endif // defined(USE_GPU)

