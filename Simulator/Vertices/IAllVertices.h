/**
 * @file IAllVertices.h
 * 
 * @ingroup Simulator/Vertices
 *
 * @brief An interface for vertices classes.
 */

#pragma once

using namespace std;

#include <iostream>

#include "Core/EdgeIndexMap.h"

class IAllEdges;

class Layout;

class IAllVertices {
public:
   virtual ~IAllVertices() {}

   ///  Setup the internal structure of the class.
   ///  Allocate memories to store all vertices' state.
   virtual void setupVertices() = 0;

   ///  Load member variables from configuration file.
   ///  Registered to OperationManager as Operation::loadParameters
   virtual void loadParameters() = 0;

   ///  Prints out all parameters of the vertices to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const = 0;

   ///  Creates all the Vertices and assigns initial data for them.
   ///
   ///  @param  layout      Layout information of the neural network.
   virtual void createAllVertices(Layout *layout) = 0;

   ///  Outputs state of the vertex chosen as a string.
   ///
   ///  @param  i   index of the vertex (in vertices) to output info from.
   ///  @return the complete state of the vertex.
   virtual string toString(const int i) const = 0;

#if defined(USE_GPU)
   public:
       ///  Allocate GPU memories to store all vertices' states,
       ///  and copy them from host to GPU memory.
       ///
       ///  @param  allNeuronsDevice   GPU address of the allVertices struct on device memory.
       virtual void allocNeuronDeviceStruct(void** allNeuronsDevice) = 0;

       ///  Delete GPU memories.
       ///
       ///  @param  allNeuronsDevice   GPU address of the allVertices struct on device memory.
       virtual void deleteNeuronDeviceStruct(void* allNeuronsDevice) = 0;

       ///  Copy all vertices' data from host to device.
       ///
       ///  @param  allNeuronsDevice   GPU address of the allVertices struct on device memory.
       virtual void copyNeuronHostToDevice(void* allNeuronsDevice) = 0;

       ///  Copy all vertices' data from device to host.
       ///
       ///  @param  allNeuronsDevice   GPU address of the allVertices struct on device memory.
       virtual void copyNeuronDeviceToHost(void* allNeuronsDevice) = 0;

       ///  Update the state of all vertices for a time step
       ///  Notify outgoing synapses if vertex has fired.
       ///
       ///  @param  synapses               Reference to the allSynapses struct on host memory.
       ///  @param  allNeuronsDevice       GPU address of the allVertices struct on device memory.
       ///  @param  allSynapsesDevice      GPU address of the allSynapses struct on device memory.
       ///  @param  randNoise              Reference to the random noise array.
       ///  @param  synapseIndexMapDevice  GPU address of the EdgeIndexMap on device memory.
       virtual void advanceVertices(IAllEdges &synapses, void* allNeuronsDevice, void* allSynapsesDevice, float* randNoise, EdgeIndexMap* synapseIndexMapDevice) = 0;

       ///  Set some parameters used for advanceVerticesDevice.
       ///
       ///  @param  synapses               Reference to the allSynapses struct on host memory.
       virtual void setAdvanceVerticesDeviceParams(IAllEdges &synapses) = 0;
#else // !defined(USE_GPU)
public:
   ///  Update internal state of the indexed Neuron (called by every simulation step).
   ///  Notify outgoing synapses if vertex has fired.
   ///
   ///  @param  synapses         The Synapse list to search from.
   ///  @param  edgeIndexMap  Reference to the EdgeIndexMap.
   virtual void advanceVertices(IAllEdges &synapses, const EdgeIndexMap *edgeIndexMap) = 0;

#endif // defined(USE_GPU)
};
