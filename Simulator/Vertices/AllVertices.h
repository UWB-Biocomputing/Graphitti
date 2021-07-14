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

#include <iostream>
#include <log4cplus/loggingmacros.h>

#include "BGTypes.h"
#include "Simulator.h"
#include "Core/EdgeIndexMap.h"
#include "AllEdges.h"
#include "Layout.h"

class Layout;
class AllEdges;

class AllVertices {
public:
   AllVertices();

   virtual ~AllVertices();

   ///  Setup the internal structure of the class.
   ///  Allocate memories to store all neurons' state.
   virtual void setupVertices();

   ///  Prints out all parameters of the neurons to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const;

   ///  Load member variables from configuration file.
   ///  Registered to OperationManager as Operation::loadParameters
   virtual void loadParameters() = 0;

   ///  Creates all the Vertices and assigns initial data for them.
   ///
   ///  @param  layout      Layout information of the neural network.
   virtual void createAllVertices(Layout *layout) = 0;

   ///  Outputs state of the vertex chosen as a string.
   ///
   ///  @param  i   index of the vertex (in vertices) to output info from.
   ///  @return the complete state of the vertex.
   virtual string toString(const int i) const = 0;

   ///  The summation point for each vertex.
   ///  Summation points are places where the synapses connected to the vertex
   ///  apply (summed up) their PSRs (Post-Synaptic-Response).
   ///  On the next advance cycle, vertices add the values stored in their corresponding
   ///  summation points to their Vm and resets the summation points to zero
   BGFLOAT *summationMap_;

protected:
   ///  Total number of vertices.
   int size_;

   // Loggers used to print to using log4cplus logging macros
   log4cplus::Logger fileLogger_; // Logs to Output/Debug/logging.txt
   log4cplus::Logger vertexLogger_; // Logs to Output/Debug/neurons.txt

#if defined(USE_GPU)
   public:
       ///  Allocate GPU memories to store all vertices' states,
       ///  and copy them from host to GPU memory.
       ///
       ///  @param  allVerticesDevice   GPU address of the allVertices struct on device memory.
       virtual void allocNeuronDeviceStruct(void** allVerticesDevice) = 0;

       ///  Delete GPU memories.
       ///
       ///  @param  allVerticesDevice   GPU address of the allVertices struct on device memory.
       virtual void deleteNeuronDeviceStruct(void* allVerticesDevice) = 0;

       ///  Copy all vertices' data from host to device.
       ///
       ///  @param  allVerticesDevice   GPU address of the allVertices struct on device memory.
       virtual void copyNeuronHostToDevice(void* allVerticesDevice) = 0;

       ///  Copy all vertices' data from device to host.
       ///
       ///  @param  allVerticesDevice   GPU address of the allVertices struct on device memory.
       virtual void copyNeuronDeviceToHost(void* allVerticesDevice) = 0;

       ///  Update the state of all vertices for a time step
       ///  Notify outgoing synapses if vertex has fired.
       ///
       ///  @param  edges               Reference to the allEdges struct on host memory.
       ///  @param  allVerticesDevice       GPU address of the allVertices struct on device memory.
       ///  @param  allEdgesDevice      GPU address of the allEdges struct on device memory.
       ///  @param  randNoise              Reference to the random noise array.
       ///  @param  edgeIndexMapDevice  GPU address of the EdgeIndexMap on device memory.
       virtual void advanceVertices(AllEdges &edges, void* allVerticesDevice, void* allEdgesDevice, float* randNoise, EdgeIndexMap* edgeIndexMapDevice) = 0;

       ///  Set some parameters used for advanceVerticesDevice.
       ///
       ///  @param  edges               Reference to the allEdges struct on host memory.
       virtual void setAdvanceVerticesDeviceParams(AllEdges &edges) = 0;
#else // !defined(USE_GPU)
public:
   ///  Update internal state of the indexed Neuron (called by every simulation step).
   ///  Notify outgoing synapses if vertex has fired.
   ///
   ///  @param  edges         The Synapse list to search from.
   ///  @param  edgeIndexMap  Reference to the EdgeIndexMap.
   virtual void advanceVertices(AllEdges &edges, const EdgeIndexMap *edgeIndexMap) = 0;

#endif // defined(USE_GPU)
};

#if defined(USE_GPU)
struct AllVerticesDeviceProperties
{
        ///  The summation point for each vertex.
        ///  Summation points are places where the synapses connected to the vertex 
        ///  apply (summed up) their PSRs (Post-Synaptic-Response). 
        ///  On the next advance cycle, vertices add the values stored in their corresponding 
        ///  summation points to their Vm and resets the summation points to zero
        BGFLOAT *summationMap_;
};
#endif // defined(USE_GPU)
