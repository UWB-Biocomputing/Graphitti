/**
 * @file AllEdges.h
 * 
 * @ingroup Simulator/Edges
 *
 * @brief An interface and top level implementation for edge classes.
 */

#pragma once

#include <vector>

#include "Global.h"
#include "Simulator.h"
#include "EdgeIndexMap.h"
#include "log4cplus/loggingmacros.h"
#include "cereal/types/vector.hpp"

class AllVertices;

class AllEdges {
public:
   AllEdges();
   AllEdges(const int numVertices, const int maxEdges);
   virtual ~AllEdges();

   ///  Setup the internal structure of the class (allocate memories and initialize them).
   virtual void setupEdges();

   /// Load member variables from configuration file.
   /// Registered to OperationManager as Operation::op::loadParameters
   virtual void loadParameters();

   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const;

   ///  Adds a Edge to the model, connecting two Vertices.
   ///
   ///  @param  iEdg        Index of the edge to be added.
   ///  @param  type        The type of the Edge to add.
   ///  @param  srcVertex   The Vertex that sends to this Edge.
   ///  @param  destVertex  The Vertex that receives from the Edge.
   ///  @param  sumPoint    Summation point address.
   ///  @param  deltaT      Inner simulation step duration
   virtual void
   addEdge(BGSIZE &iEdg, edgeType type, const int srcVertex, const int destVertex, BGFLOAT *sumPoint,
              const BGFLOAT deltaT);

   ///  Create a Edge and connect it to the model.
   ///
   ///  @param  iEdg        Index of the edge to set.
   ///  @param  srcVertex   Coordinates of the source Vertex.
   ///  @param  destVertex  Coordinates of the destination Vertex.
   ///  @param  sumPoint    Summation point address.
   ///  @param  deltaT      Inner simulation step duration.
   ///  @param  type        Type of the Edge to create.
   virtual void createEdge(const BGSIZE iEdg, int srcVertex, int destVertex, BGFLOAT *sumPoint, const BGFLOAT deltaT,
                              edgeType type) = 0;

   ///  Populate a edge index map.
   virtual void createEdgeIndexMap(shared_ptr<EdgeIndexMap> edgeIndexMap);

   ///  Cereal serialization method
   ///  (Serializes edge weights, source vertices, and destination vertices)
   template<class Archive>
   void save(Archive &archive) const;

   ///  Cereal deserialization method
   ///  (Deserializes edge weights, source vertices, and destination vertices)
   template<class Archive>
   void load(Archive &archive);

protected:
   ///  Setup the internal structure of the class (allocate memories and initialize them).
   ///
   ///  @param  numVertices   Total number of vertices in the network.
   ///  @param  maxEdges  Maximum number of edges per vertex.
   virtual void setupEdges(const int numVertices, const int maxEdges);

   ///  Sets the data for Edge to input's data.
   ///
   ///  @param  input  istream to read from.
   ///  @param  iEdg   Index of the edge to set.
   virtual void readEdge(istream &input, const BGSIZE iEdg);

   ///  Write the edge data to the stream.
   ///
   ///  @param  output  stream to print out to.
   ///  @param  iEdg    Index of the edge to print out.
   virtual void writeEdge(ostream &output, const BGSIZE iEdg) const;
   
   ///  Returns an appropriate edgeType object for the given integer.
   ///
   ///  @param  typeOrdinal    Integer that correspond with a edgeType.
   ///  @return the SynapseType that corresponds with the given integer.
   edgeType edgeOrdinalToType(const int typeOrdinal);

   /// Loggers used to print to using log4cplus logging macros, prints to Results/Debug/logging.txt
   log4cplus::Logger fileLogger_;
   log4cplus::Logger edgeLogger_;
 
#if defined(USE_GPU)
   public:
       ///  Allocate GPU memories to store all edges' states,
       ///  and copy them from host to GPU memory.
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       virtual void allocEdgeDeviceStruct(void** allEdgesDevice) = 0;

       ///  Allocate GPU memories to store all edges' states,
       ///  and copy them from host to GPU memory.
       ///
       ///  @param  allEdgesDevice     GPU address of the allEdges struct on device memory.
       ///  @param  numVertices            Number of vertices.
       ///  @param  maxEdgesPerVertex  Maximum number of edges per vertex.
       virtual void allocEdgeDeviceStruct( void** allEdgesDevice, int numVertices, int maxEdgesPerVertex ) = 0;

       ///  Delete GPU memories.
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       virtual void deleteEdgeDeviceStruct( void* allEdgesDevice ) = 0;

       ///  Copy all edges' data from host to device.
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       virtual void copyEdgeHostToDevice(void* allEdgesDevice) = 0;

       ///  Copy all edges' data from host to device.
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       ///  @param  numVertices           Number of vertices.
       ///  @param  maxEdgesPerVertex  Maximum number of edges per vertex.
       virtual void copyEdgeHostToDevice( void* allEdgesDevice, int numVertices, int maxEdgesPerVertex ) = 0;

       ///  Copy all edges' data from device to host.
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       virtual void copyEdgeDeviceToHost( void* allEdgesDevice) = 0;

       ///  Get edge_counts in AllEdges struct on device memory.
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       virtual void copyDeviceEdgeCountsToHost(void* allEdgesDevice) = 0;

       ///  Get summationCoord and in_use in AllEdges struct on device memory.
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       virtual void copyDeviceEdgeSumIdxToHost(void* allEdgesDevice) = 0;

       ///  Advance all the Synapses in the simulation.
       ///  Update the state of all edges for a time step.
       ///
       ///  @param  allEdgesDevice      GPU address of the allEdges struct on device memory.
       ///  @param  allVerticesDevice   GPU address of the allNeurons struct on device memory.
       ///  @param  edgeIndexMapDevice  GPU address of the EdgeIndexMap on device memory.
       virtual void advanceEdges(void* allEdgesDevice, void* allVerticesDevice, void* edgeIndexMapDevice) = 0;

       ///  Set some parameters used for advanceEdgesDevice.
       virtual void setAdvanceEdgesDeviceParams() = 0;

       ///  Set edge class ID defined by enumClassSynapses for the caller's Edge class.
       ///  The class ID will be set to classSynapses_d in device memory,
       ///  and the classSynapses_d will be referred to call a device function for the
       ///  particular edge class.
       ///  Because we cannot use virtual function (Polymorphism) in device functions,
       ///  we use this scheme.
       ///  Note: we used to use a function pointer; however, it caused the growth_cuda crash
       ///  (see issue#137).
       virtual void setEdgeClassID() = 0;

       ///  Prints GPU SynapsesProps data.
       ///
       ///  @param  allEdgesDeviceProps   GPU address of the corresponding SynapsesDeviceProperties struct on device memory.
       virtual void printGPUEdgesProps( void* allEdgesDeviceProps ) const = 0;

#else // !defined(USE_GPU)
public:
   ///  Advance all the edges in the simulation.
   ///  Update the state of all edges for a time step.
   ///
   ///  @param  vertices       The Vertex list to search from.
   ///  @param  edgeIndexMap   Pointer to EdgeIndexMap structure.
   virtual void advanceEdges(AllVertices *vertices, EdgeIndexMap *edgeIndexMap);

   ///  Advance one specific Edge.
   ///
   ///  @param  iEdg      Index of the Edge to connect to.
   ///  @param  vertices  The Vertex list to search from.
   virtual void advanceEdge(const BGSIZE iEdg, AllVertices *vertices) = 0;

   ///  Remove a edge from the network.
   ///
   ///  @param  neuronIndex   Index of a vertex to remove from.
   ///  @param  iEdg          Index of a edge to remove.
   virtual void eraseEdge(const int neuronIndex, const BGSIZE iEdg);
#endif // defined(USE_GPU)

   ///  The location of the edge.
   int *sourceVertexIndex_;

   ///  The coordinates of the summation point.
   int *destVertexIndex_;

   ///   The weight (scaling factor, strength, maximal amplitude) of the edge.
   BGFLOAT *W_;

   ///  This edge's summation point's address.
   BGFLOAT **summationPoint_;

   ///   Synapse type
   edgeType *type_;

   ///  The boolean value indicating the entry in the array is in use.
   bool *inUse_;

   ///  The number of (incoming) edges for each vertex.
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

///  Cereal serialization method
///  (Serializes edge weights, source vertices, and destination vertices)
template<class Archive>
void AllEdges::save(Archive &archive) const {
   // uses vector to save edge weights, source vertices, and destination vertices
   vector<BGFLOAT> WVector;
   vector<int> sourceVertexLayoutIndexVector;
   vector<int> destVertexLayoutIndexVector;

   for (int i = 0; i < maxEdgesPerVertex_ * countVertices_; i++) {
      WVector.push_back(W_[i]);
      sourceVertexLayoutIndexVector.push_back(sourceVertexIndex_[i]);
      destVertexLayoutIndexVector.push_back(destVertexIndex_[i]);
   }

   // serialization
   archive(WVector, sourceVertexLayoutIndexVector, destVertexLayoutIndexVector);
}

///  Cereal deserialization method
///  (Deserializes edge weights, source vertices, and destination vertices)
template<class Archive>
void AllEdges::load(Archive &archive) {
   // uses vectors to load edge weights, source vertices, and destination vertices
   vector<BGFLOAT> WVector;
   vector<int> sourceVertexLayoutIndexVector;
   vector<int> destVertexLayoutIndexVector;

   // deserializing data to these vectors
   archive(WVector, sourceVertexLayoutIndexVector, destVertexLayoutIndexVector);

   // check to see if serialized data sizes matches object sizes
   if (WVector.size() != maxEdgesPerVertex_ * countVertices_) {
      cerr
            << "Failed deserializing edge weights, source vertices, and/or destination vertices. Please verify maxEdgesPerVertex and count_neurons data members in AllEdges class."
            << endl;
      throw cereal::Exception("Deserialization Error");
   }

   // assigns serialized data to objects
   for (int i = 0; i < maxEdgesPerVertex_ * countVertices_; i++) {
      W_[i] = WVector[i];
      sourceVertexIndex_[i] = sourceVertexLayoutIndexVector[i];
      destVertexIndex_[i] = destVertexLayoutIndexVector[i];
   }
}
