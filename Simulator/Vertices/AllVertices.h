/**
 * @file AllVertices.h
 * 
 * @ingroup Simulator/Vertices
 *
 * @brief A container of the base class of all vertex data
 *
 * The class uses a data-centric structure, which utilizes a structure as the containers of
 * all vertices.
 *
 * The container holds vertex parameters of all vertices.
 * Each kind of vertex parameter is stored in a 1D array, of which length
 * is number of all vertices. Each array of a vertex parameter is pointed by a
 * corresponding member variable of the vertex parameter in the class.
 *
 * This structure was originally designed for the GPU implementation of the
 * simulator, and this refactored version of the simulator simply uses that design for
 * all other implementations as well. This is to simplify transitioning from
 * single-threaded to multi-threaded.
 */

#pragma once

using namespace std;

#include "AllEdges.h"
#include "BGTypes.h"
#include "Core/EdgeIndexMap.h"
#include "Layout.h"
#include "Simulator.h"
#include <iostream>
#include <log4cplus/loggingmacros.h>
// cereal
#include "cereal/types/vector.hpp"
#if defined(USE_GPU)
   #include <cuda_runtime.h>
#endif

// Utility function to convert a vertexType into a string.
string vertexTypeToString(vertexType t);

class Layout;
class AllEdges;
struct AllVerticesDeviceProperties;

class AllVertices {
public:
   AllVertices();

   virtual ~AllVertices() = default;

   ///  Setup the internal structure of the class.
   ///  Allocate memories to store all vertices' state.
   virtual void setupVertices();

   ///  Prints out all parameters of the vertices to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const;

   /// Loads all inputs scheduled to occur in the upcoming epoch.
   /// These are inputs occurring in between curStep (inclusive) and
   /// endStep (exclusive)
   ///
   /// @param  curStep  The current simulation step
   /// @param  endStep  The end of epoch simulation step
   virtual void loadEpochInputs(uint64_t currentStep, uint64_t endStep);

   ///  Load member variables from configuration file.
   ///  Registered to OperationManager as Operation::loadParameters
   virtual void loadParameters() = 0;

   ///  Creates all the Vertices and assigns initial data for them.
   ///
   ///  @param  layout      Layout information of the network.
   virtual void createAllVertices(Layout &layout) = 0;

   ///  Outputs state of the vertex chosen as a string.
   ///
   ///  @param  i   index of the vertex (in vertices) to output info from.
   ///  @return the complete state of the vertex.
   virtual string toString(int i) const = 0;

   /// Helper function for recorder to register spike history variables for all vertices.
   /// Option 1: Register vertex information in vertexEvents_ one by one.
   /// Option 2: Register a vector of EventBuffer variables.
   virtual void registerHistoryVariables() = 0;

   ///  Cereal serialization method
   template <class Archive> void serialize(Archive &archive);

protected:
   ///  Total number of vertices.
   int size_;

   // Loggers used to print to using log4cplus logging macros
   log4cplus::Logger fileLogger_;   // Logs to Output/Debug/logging.txt
   /// TODO: Should check to see if this file name "neurons" is hardcoded and change it to vertices
   log4cplus::Logger vertexLogger_;   // Logs to Output/Debug/neurons.txt

#if defined(USE_GPU)
   ///  Cuda Stream for Edge Kernels
   cudaStream_t simulationStream_;

public:
   /// Set the CUDA stream to be used by GPU vertices kernels in derived classes.
   ///
   /// This assigns a CUDA stream to the base class, allowing subclasses
   /// to launch kernels on the correct stream. The stream is typically
   /// created by GPUModel and passed down during simulation setup.
   ///
   /// @param simulationStream A valid CUDA stream (`cudaStream_t`) managed by the caller.
   void SetStream(cudaStream_t simulationStream);

   ///  Allocate GPU memories to store all vertices' states,
   ///  and copy them from host to GPU memory.
   virtual void allocVerticesDeviceStruct() = 0;

   ///  Delete GPU memories.
   ///
   virtual void deleteVerticesDeviceStruct() = 0;

   ///  Clear the spike counts out of all vertices.
   //
   ///  @param  allVerticesDevice   GPU address of the allVertices struct on device memory.
   virtual void clearVertexHistory(void *allVerticesDevice) = 0;

   ///  Copy all vertices' data from host to device.
   virtual void copyToDevice() = 0;

   ///  Copy all vertices' data from device to host.
   ///
   virtual void copyFromDevice() = 0;

   ///  Update the state of all vertices for a time step
   ///  Notify outgoing edges if vertex has fired.
   ///
   ///  @param  edges               Reference to the allEdges struct on host memory.
   ///  @param  allVerticesDevice       GPU address of the allVertices struct on device memory.
   ///  @param  allEdgesDevice      GPU address of the allEdges struct on device memory.
   ///  @param  randNoise              Reference to the random noise array.
   ///  @param  edgeIndexMapDevice  GPU address of the EdgeIndexMap on device memory.
   virtual void advanceVertices(AllEdges &edges, void *allVerticesDevice, void *allEdgesDevice,
                                float randNoise[], EdgeIndexMapDevice *edgeIndexMapDevice)
      = 0;

   ///  Set some parameters used for advanceVerticesDevice.
   ///
   ///  @param  edges               Reference to the allEdges struct on host memory.
   virtual void setAdvanceVerticesDeviceParams(AllEdges &edges) = 0;

   /// Performs an integration operation per vertex using the inputs to the vertex.
   ///
   /// @param allVerticesDevice       GPU address of the allVertices struct on device memory.
   /// @param edgeIndexMapDevice      GPU address of the EdgeIndexMap on device memory.
   /// @param allEdgesDevice          GPU address of the allEdges struct on device memory.
   virtual void integrateVertexInputs(void *allVerticesDevice,
                                      EdgeIndexMapDevice *edgeIndexMapDevice, void *allEdgesDevice)
      = 0;
#else   // !defined(USE_GPU)
public:
   ///  Update internal state of the indexed vertex (called by every simulation step).
   ///  Notify outgoing edges if vertex has fired.
   ///
   ///  @param  edges         The edge list to search from.
   ///  @param  edgeIndexMap  Reference to the EdgeIndexMap.
   virtual void advanceVertices(AllEdges &edges, const EdgeIndexMap &edgeIndexMap) = 0;

   /// Performs an integration operation per vertex using the inputs to the vertex.
   ///
   ///  @param  edges         The edge list to search from.
   ///  @param  edgeIndexMap  Reference to the EdgeIndexMap.
   virtual void integrateVertexInputs(AllEdges &edges, EdgeIndexMap &edgeIndexMap) = 0;

#endif   // defined(USE_GPU)
};

#if defined(USE_GPU)
struct AllVerticesDeviceProperties {};
#endif   // defined(USE_GPU)

template <class Archive> void AllVertices::serialize(Archive &archive)
{
   archive(cereal::make_nvp("size", size_));
}