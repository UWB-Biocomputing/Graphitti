/**
 * @file AllEdges.h
 * 
 * @ingroup Simulator/Edges
 *
 * @brief An interface and top level implementation for edge classes.
 */

#pragma once

#include "EdgeIndexMap.h"
#include "Global.h"
#include "Simulator.h"
#include "log4cplus/loggingmacros.h"
#include <vector>
// cereal
#include "cereal/types/vector.hpp"
#ifdef USE_GPU
   #include <cuda_runtime.h>
#endif

class AllVertices;
struct AllEdgesDeviceProperties;

class AllEdges {
public:
   AllEdges();
   AllEdges(int numVertices, int maxEdges);
   virtual ~AllEdges() = default;

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
   ///  @param  type        The type of the Edge to add.
   ///  @param  srcVertex   The Vertex that sends to this Edge.
   ///  @param  destVertex  The Vertex that receives from the Edge.
   ///  @param  deltaT      Inner simulation step duration
   ///  @return  iEdg        Index of the edge to be added.
   virtual BGSIZE addEdge(edgeType type, int srcVertex, int destVertex, BGFLOAT deltaT);

   ///  Create a Edge and connect it to the model.
   ///
   ///  @param  iEdg        Index of the edge to set.
   ///  @param  srcVertex   Coordinates of the source Vertex.
   ///  @param  destVertex  Coordinates of the destination Vertex.
   ///  @param  deltaT      Inner simulation step duration.
   ///  @param  type        Type of the Edge to create.
   virtual void createEdge(BGSIZE iEdg, int srcVertex, int destVertex, BGFLOAT deltaT,
                           edgeType type)
      = 0;

   ///  Populate a edge index map.
   virtual void createEdgeIndexMap(EdgeIndexMap &edgeIndexMap);

   ///  Cereal serialization method
   template <class Archive> void serialize(Archive &archive);

protected:
   ///  Setup the internal structure of the class (allocate memories and initialize them).
   ///
   ///  @param  numVertices   Total number of vertices in the network.
   ///  @param  maxEdges  Maximum number of edges per vertex.
   virtual void setupEdges(int numVertices, int maxEdges);

   ///  Sets the data for Edge to input's data.
   ///
   ///  @param  input  istream to read from.
   ///  @param  iEdg   Index of the edge to set.
   virtual void readEdge(istream &input, BGSIZE iEdg);

   ///  Write the edge data to the stream.
   ///
   ///  @param  output  stream to print out to.
   ///  @param  iEdg    Index of the edge to print out.
   virtual void writeEdge(ostream &output, BGSIZE iEdg) const;

   ///  Returns an appropriate edgeType object for the given integer.
   ///
   ///  @param  typeOrdinal    Integer that correspond with a edgeType.
   ///  @return the edgeType that corresponds with the given integer.
   edgeType edgeOrdinalToType(int typeOrdinal);

   /// Loggers used to print to using log4cplus logging macros, prints to Results/Debug/logging.txt
   log4cplus::Logger fileLogger_;
   log4cplus::Logger edgeLogger_;

#if defined(USE_GPU)
   ///  Cuda Stream for Edge Kernels
   cudaStream_t simulationStream_;

public:
   /// Set the CUDA stream to be used by GPU edge kernels in derived classes.
   ///
   /// This assigns a CUDA stream to the base class, allowing subclasses
   /// (e.g., AllSpikingSynapses_d, AllSTDPSynapses_d) to launch kernels on
   /// the correct stream. The stream is typically created by GPUModel and
   /// passed down during simulation setup.
   ///
   /// @param simulationStream A valid CUDA stream (`cudaStream_t`) managed by the caller.
   void SetStream(cudaStream_t simulationStream);

   ///  Allocate GPU memories to store all edges' states,
   ///  and copy them from host to GPU memory.
   virtual void allocEdgeDeviceStruct() = 0;

   ///  Allocate GPU memories to store all edges' states,
   ///  and copy them from host to GPU memory.
   ///
   ///  @param  allEdgesDevice     GPU address of the allEdges struct on device memory.
   ///  @param  numVertices            Number of vertices.
   ///  @param  maxEdgesPerVertex  Maximum number of edges per vertex.
   virtual void allocEdgeDeviceStruct(void **allEdgesDevice, int numVertices, int maxEdgesPerVertex)
      = 0;

   ///  Delete GPU memories.
   ///
   virtual void deleteEdgeDeviceStruct() = 0;

   ///  Copy all edges' data from host to device.
   virtual void copyEdgeHostToDevice() = 0;

   ///  Copy all edges' data from host to device.
   ///
   ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
   ///  @param  numVertices           Number of vertices.
   ///  @param  maxEdgesPerVertex  Maximum number of edges per vertex.
   virtual void copyEdgeHostToDevice(void *allEdgesDevice, int numVertices, int maxEdgesPerVertex)
      = 0;

   ///  Copy all edges' data from device to host.
   ///
   virtual void copyEdgeDeviceToHost() = 0;

   ///  Get edge_counts in AllEdges struct on device memory.
   ///
   ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
   virtual void copyDeviceEdgeCountsToHost(void *allEdgesDevice) = 0;

   ///  Advance all the edges in the simulation.
   ///  Update the state of all edges for a time step.
   ///
   ///  @param  allEdgesDevice      GPU address of the allEdges struct on device memory.
   ///  @param  allVerticesDevice   GPU address of the allVertices struct on device memory.
   ///  @param  edgeIndexMapDevice  GPU address of the EdgeIndexMap on device memory.
   virtual void advanceEdges(void *allEdgesDevice, void *allVerticesDevice,
                             void *edgeIndexMapDevice)
      = 0;

   ///  Set some parameters used for advanceEdgesDevice.
   virtual void setAdvanceEdgesDeviceParams() = 0;

   ///  TODO: Clean up this comment to remove synapses reference since this is neuro-specific
   ///  Set edge class ID defined by enumClassSynapses for the caller's Edge class.
   ///  The class ID will be set to classSynapses_d in device memory,
   ///  and the classSynapses_d will be referred to call a device function for the
   ///  particular edge class.
   ///  Because we cannot use virtual function (Polymorphism) in device functions,
   ///  we use this scheme.
   ///  Note: we used to use a function pointer; however, it caused the growth_cuda crash
   ///  (see issue#137).
   virtual void setEdgeClassID() = 0;

   ///  Prints GPU edgesProps data.
   ///
   ///  @param  allEdgesDeviceProps   GPU address of the corresponding AllEdgesDeviceProperties struct on device memory.
   virtual void printGPUEdgesProps(void *allEdgesDeviceProps) const = 0;

#else    // !defined(USE_GPU)
public:
   ///  Advance all the edges in the simulation.
   ///  Update the state of all edges for a time step.
   ///
   ///  @param  vertices       The Vertex list to search from.
   ///  @param  edgeIndexMap   Pointer to EdgeIndexMap structure.
   virtual void advanceEdges(AllVertices &vertices, EdgeIndexMap &edgeIndexMap);

   ///  Advance one specific Edge.
   ///
   ///  @param  iEdg      Index of the Edge to connect to.
   ///  @param  vertices  The Vertex list to search from.
   virtual void advanceEdge(BGSIZE iEdg, AllVertices &vertices) = 0;

   ///  Remove a edge from the network.
   ///
   ///  @param  vertexIndex   Index of a vertex to remove from.
   ///  @param  iEdg          Index of a edge to remove.
   virtual void eraseEdge(int vertexIndex, BGSIZE iEdg);
#endif   // defined(USE_GPU)

   ///  The location of the edge.
   vector<int> sourceVertexIndex_;

   ///  TODO: Should generalize this comment since summation point is neuro-specific
   ///  The coordinates of the summation point.
   vector<int> destVertexIndex_;

   ///   The weight (scaling factor, strength, maximal amplitude) of the edge.
   vector<BGFLOAT> W_;

   ///   edge type
   vector<edgeType> type_;

   ///  The value indicating the entry in the array is in use.
   // The representation of inUse has been updated from bool to unsigned char
   // to store 1 (true) or 0 (false) for the support of serialization operations. See ISSUE-459
   vector<unsigned char> inUse_;

   ///  The number of (incoming) edges for each vertex.
   ///  Note: Likely under a different name in GpuSim_struct, see edge_count. -Aaron
   vector<BGSIZE> edgeCounts_;

   ///  The total number of active edges.
   BGSIZE totalEdgeCount_;

   ///  The maximum number of edges for each vertex.
   BGSIZE maxEdgesPerVertex_;

   ///  The number of vertices
   ///  Aaron: Is this even supposed to be here?!
   ///  Usage: Used by destructor
   int countVertices_;
};

#if defined(USE_GPU)
struct AllEdgesDeviceProperties {
   ///  The location of the edge.
   int *sourceVertexIndex_;

   ///  TODO: Should generalize this comment since summation point is neuro-specific
   ///  The coordinates of the summation point.
   int *destVertexIndex_;

   ///   The weight (scaling factor, strength, maximal amplitude) of the edge.
   BGFLOAT *W_;

   ///  edge type
   edgeType *type_;

   ///  The value indicating the entry in the array is in use.
   // The representation of inUse has been updated from bool to unsigned char
   // to store 1 (true) or 0 (false) for the support of serialization operations. See ISSUE-459
   unsigned char *inUse_;

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
#endif   // defined(USE_GPU)

///  Cereal serialization method
///  (Serializes edge weights, source vertices, and destination vertices)
template <class Archive> void AllEdges::serialize(Archive &archive)
{
   // serialization
   archive(cereal::make_nvp("sourceVertexIndex", sourceVertexIndex_),
           cereal::make_nvp("edgeWeights", W_),
           cereal::make_nvp("destVertexIndex", destVertexIndex_), cereal::make_nvp("type", type_),
           cereal::make_nvp("inUse", inUse_), cereal::make_nvp("edgeCounts", edgeCounts_),
           cereal::make_nvp("totalEdgeCount", totalEdgeCount_),
           cereal::make_nvp("maxEdgesPerVertex", maxEdgesPerVertex_),
           cereal::make_nvp("countVertices", countVertices_));
}