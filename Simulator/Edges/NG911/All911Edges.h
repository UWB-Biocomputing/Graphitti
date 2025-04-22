/**
 * @file All911Edges.h
 *
 * @ingroup Simulator/Edges/NG911
 *
 * @brief Specialization of the AllEdges class for the NG911 network
 * 
 * In the NG911 Model, an edge represent a communication link between two nodes.
 * When communication messages, such as calls, are send between the vertices;
 * they are placed in the outgoing edge of the vertex where the message
 * originates. These messages are then pulled, for processing, by the destination
 * vertex.
 * 
 * Besides having a placeholder for the message being sent, each edge has
 * parameters to keep track of its availability. It is worth mentioning that
 * because the class contains all the edges; these parameters are contained in
 * vectors or arrays, where each item correspond to an edge parameter.
 * 
 * During the `advanceEdges` step of the simulation the destination vertex pulls
 * the calls placed in the given edge and queues them into their internal queue.
 * Here we check that there is space in the queue, the queue in considered full
 * if all trunks are busy. That is, if the total calls in the waiting queue +
 * the number of busy agents equals the total number of trunks available.
 */

#pragma once

#include "All911Vertices.h"
#include "AllEdges.h"


struct All911EdgeDeviceProperties;

class All911Edges : public AllEdges {
public:
   All911Edges() = default;

   All911Edges(int numVertices, int maxEdges);

   virtual ~All911Edges() = default;

   ///  Creates an instance of the class.
   ///
   ///  @return Reference to the instance of the class.
   static AllEdges *Create()
   {
      return new All911Edges();
   }

   ///  Setup the internal structure of the class (allocate memories and initialize them).
   virtual void setupEdges() override;

   ///  Create a Edge and connect it to the model.
   ///
   ///  @param  iEdg        Index of the edge to set.
   ///  @param  srcVertex   Coordinates of the source Vertex.
   ///  @param  destVertex  Coordinates of the destination Vertex.
   ///  @param  deltaT      Inner simulation step duration.
   ///  @param  type        Type of the Edge to create.
   virtual void createEdge(BGSIZE iEdg, int srcVertex, int destVertex, BGFLOAT deltaT,
                           edgeType type) override;

protected:
#if defined(USE_GPU)
   // GPU functionality for 911 simulation is unimplemented.
   // These signatures are required to make the class non-abstract
public:
   virtual void allocEdgeDeviceStruct() {};
   virtual void allocEdgeDeviceStruct(void **allEdgesDevice, int numVertices,
                                      int maxEdgesPerVertex) {};
   virtual void deleteEdgeDeviceStruct() {};
   virtual void copyEdgeHostToDevice() {};
   virtual void copyEdgeHostToDevice(void *allEdgesDevice, int numVertices, int maxEdgesPerVertex) {
   };
   virtual void copyEdgeDeviceToHost() {};
   virtual void copyDeviceEdgeCountsToHost(void *allEdgesDevice) {};
   virtual void copyDeviceEdgeSumIdxToHost(void *allEdgesDevice) {};
   virtual void advanceEdges(void *allEdgesDevice, void *allVerticesDevice,
                             void *edgeIndexMapDevice) {};
   virtual void setAdvanceEdgesDeviceParams() {};
   virtual void setEdgeClassID() {};
   virtual void printGPUEdgesProps(void *allEdgesDeviceProps) const {};

#else   // !defined(USE_GPU)
public:
   ///  Advance all the edges in the simulation.
   ///
   ///  @param  vertices       The vertex list to search from.
   ///  @param  edgeIndexMap   Pointer to EdgeIndexMap structure.
   virtual void advanceEdges(AllVertices &vertices, EdgeIndexMap &edgeIndexMap);

   ///  Advance one specific Edge.
   ///
   ///  @param  iEdg      Index of the Edge to connect to.
   ///  @param  vertices  The Neuron list to search from.
   void advance911Edge(BGSIZE iEdg, All911Vertices &vertices);

   /// unused virtual function placeholder
   virtual void advanceEdge(BGSIZE iEdg, AllVertices &vertices) override {};

#endif

   /// If edge has a call or not
   unique_ptr<bool[]> isAvailable_;

   /// If the call in the edge is a redial
   unique_ptr<bool[]> isRedial_;

   /// The call information per edge
   vector<Call> call_;
};
