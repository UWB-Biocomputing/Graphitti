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
#include "Global.h"
// cereal
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>

struct All911EdgesDeviceProperties;

// enumerate all non-abstract edge classes.
enum ESCSEdges {
   NineOneOneEdges,
   undefESCSEdges
};

enum ResponderTypes {
   EMS = vertexType::EMS,
   FIRE = vertexType::FIRE,
   LAW = vertexType::LAW
};

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
   
   ///  Cereal serialization method
   template <class Archive> void serialize(Archive &archive);

protected:
#if defined(USE_GPU)
   // GPU functionality for 911 simulation is unimplemented.
   // These signatures are required to make the class non-abstract
public:
   virtual void allocEdgeDeviceStruct(void **allEdgesDevice) {};
   virtual void allocEdgeDeviceStruct(void **allEdgesDevice, int numVertices,
                                      int maxEdgesPerVertex) {};
   virtual void deleteEdgeDeviceStruct(void *allEdgesDevice) {};
   virtual void copyEdgeHostToDevice(void *allEdgesDevice) {};
   virtual void copyEdgeHostToDevice(void *allEdgesDevice, int numVertices,
                                     int maxEdgesPerVertex) {};
   virtual void copyEdgeDeviceToHost(void *allEdgesDevice) {};
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
   vector<unsigned char> isAvailable_;

   /// If the call in the edge is a redial
   vector<unsigned char> isRedial_;

   /// The call information per edge
   //
   // The vertexId where the input event happen
   vector<int> vertexId_;

   // The start of the event since the beggining of
   // the simulation in timesteps matches g_simulationStep type
   vector<uint64_t> time_;

   // The duration of the event in timesteps
   vector<int> duration_;

   // Event location
   vector<double> x_;
   vector<double> y_;

   // Patience time: How long a customer is willing to wait in the queue
   vector<int> patience_;

   // On Site Time: Time spent by a responder at the site of the incident
   vector<int> onSiteTime_;
   vector<ResponderTypes> type_;
};

#if defined(USE_GPU)
struct All911EdgesDeviceProperties : public AllEdgesDeviceProperties {
   /// If edge has a call or not
   unsigned char *isAvailable_;

   /// If the call in the edge is a redial
   unsigned char *isRedial_;

   /// The call information per edge
   //
   // The vertexId where the input event happen
   int *vertexId_;

   // The start of the event since the beggining of
   // the simulation in timesteps matches g_simulationStep type
   uint64_t *time_;

   // The duration of the event in timesteps
   int *duration_;

   // Event location
   double *x_;
   double *y_;

   // Patience time: How long a customer is willing to wait in the queue
   int *patience_;

   // On Site Time: Time spent by a responder at the site of the incident
   int *onSiteTime_;
   ResponderTypes *responderType_;
};
#endif   // defined(USE_GPU)

CEREAL_REGISTER_TYPE(All911Edges);

///  Cereal serialization method
template <class Archive> void All911Edges::serialize(Archive &archive)
{
   archive(cereal::base_class<AllEdges>(this), 
           cereal::make_nvp("isAvailable", isAvailable_),
           cereal::make_nvp("isRedial", isRedial_),
           cereal::make_nvp("vertexId", vertexId_),
           cereal::make_nvp("time", time_),
           cereal::make_nvp("duration", duration_),
           cereal::make_nvp("x", x_),
           cereal::make_nvp("y", y_),
           cereal::make_nvp("patience", patience_),
           cereal::make_nvp("onSiteTime", onSiteTime_),
           cereal::make_nvp("responderType", responderType_));
}
