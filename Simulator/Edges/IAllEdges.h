/**
 * @file IAllEdges.h
 * 
 * @ingroup Simulator/Edges
 *
 * @brief An interface for edge classes.
 */

#pragma once

#include "Global.h"
#include "Core/Simulator.h"
#include "Simulator/Core/EdgeIndexMap.h"

class IAllVertices;

typedef void (*fpCreateSynapse_t)(void*, const int, const int, int, int, BGFLOAT*, const BGFLOAT, synapseType);

// enumerate all non-abstract edge classes.
enum enumClassSynapses {classAllSpikingSynapses, classAllDSSynapses, classAllSTDPSynapses, classAllDynamicSTDPSynapses, undefClassSynapses};

class IAllEdges {
public:
   virtual ~IAllEdges() {};

   ///  Setup the internal structure of the class (allocate memories and initialize them).
   virtual void setupEdges() = 0;

   ///  Reset time varying state vars and recompute decay.
   ///
   ///  @param  iEdg     Index of the edge to set.
   ///  @param  deltaT   Inner simulation step duration
   virtual void resetEdge(const BGSIZE iEdg, const BGFLOAT deltaT) = 0;

   /// Load member variables from configuration file.
   /// Registered to OperationManager as Operation::op::loadParameters
   virtual void loadParameters() = 0;

   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const = 0;

   ///  Adds a Edge to the model, connecting two Vertices.
   ///
   ///  @param  iEdg        Index of the edge to be added.
   ///  @param  type        The type of the Edge to add.
   ///  @param  srcNeuron  The Vertex that sends to this Edge.
   ///  @param  destNeuron The Vertex that receives from the Edge.
   ///  @param  sumPoint   Summation point address.
   ///  @param  deltaT      Inner simulation step duration
   virtual void
   addEdge(BGSIZE &iEdg, synapseType type, const int srcNeuron, const int destNeuron, BGFLOAT *sumPoint,
              const BGFLOAT deltaT) = 0;

   ///  Create a Edge and connect it to the model.
   ///
   ///  @param  iEdg        Index of the edge to set.
   ///  @param  srcNeuron      Coordinates of the source Vertex.
   ///  @param  destNeuron        Coordinates of the destination Vertex.
   ///  @param  sumPoint   Summation point address.
   ///  @param  deltaT      Inner simulation step duration.
   ///  @param  type        Type of the Edge to create.
   virtual void createEdge(const BGSIZE iEdg, int srcNeuron, int destNeuron, BGFLOAT *sumPoint, const BGFLOAT deltaT,
                              synapseType type) = 0;

   ///  Create a edge index map.
   virtual EdgeIndexMap *createSynapseIndexMap() = 0;

   ///  Get the sign of the synapseType.
   ///
   ///  @param    type    synapseType I to I, I to E, E to I, or E to E
   ///  @return   1 or -1, or 0 if error
   virtual int synSign(const synapseType type) = 0;

   ///  Prints SynapsesProps data to console.
   virtual void printSynapsesProps() const = 0;

#if defined(USE_GPU)
   public:
       ///  Allocate GPU memories to store all edges' states,
       ///  and copy them from host to GPU memory.
       ///
       ///  @param  allSynapsesDevice  GPU address of the allSynapses struct on device memory.
       virtual void allocSynapseDeviceStruct(void** allSynapsesDevice) = 0;

       ///  Allocate GPU memories to store all edges' states,
       ///  and copy them from host to GPU memory.
       ///
       ///  @param  allSynapsesDevice     GPU address of the allSynapses struct on device memory.
       ///  @param  numNeurons            Number of vertices.
       ///  @param  maxEdgesPerVertex  Maximum number of edges per vertex.
       virtual void allocSynapseDeviceStruct( void** allSynapsesDevice, int numNeurons, int maxEdgesPerVertex ) = 0;

       ///  Delete GPU memories.
       ///
       ///  @param  allSynapsesDevice  GPU address of the allSynapses struct on device memory.
       virtual void deleteSynapseDeviceStruct( void* allSynapsesDevice ) = 0;

       ///  Copy all edges' data from host to device.
       ///
       ///  @param  allSynapsesDevice  GPU address of the allSynapses struct on device memory.
       virtual void copySynapseHostToDevice(void* allSynapsesDevice) = 0;

       ///  Copy all edges' data from host to device.
       ///
       ///  @param  allSynapsesDevice  GPU address of the allSynapses struct on device memory.
       ///  @param  numNeurons           Number of vertices.
       ///  @param  maxEdgesPerVertex  Maximum number of edges per vertex.
       virtual void copySynapseHostToDevice( void* allSynapsesDevice, int numNeurons, int maxEdgesPerVertex ) = 0;

       ///  Copy all edges' data from device to host.
       ///
       ///  @param  allSynapsesDevice  GPU address of the allSynapses struct on device memory.
       virtual void copySynapseDeviceToHost( void* allSynapsesDevice) = 0;

       ///  Get synapse_counts in AllEdges struct on device memory.
       ///
       ///  @param  allSynapsesDevice  GPU address of the allSynapses struct on device memory.
       virtual void copyDeviceSynapseCountsToHost(void* allSynapsesDevice) = 0;

       ///  Get summationCoord and in_use in AllEdges struct on device memory.
       ///
       ///  @param  allSynapsesDevice  GPU address of the allSynapses struct on device memory.
       virtual void copyDeviceSynapseSumIdxToHost(void* allSynapsesDevice) = 0;

       ///  Advance all the Synapses in the simulation.
       ///  Update the state of all edges for a time step.
       ///
       ///  @param  allSynapsesDevice      GPU address of the allSynapses struct on device memory.
       ///  @param  allNeuronsDevice       GPU address of the allNeurons struct on device memory.
       ///  @param  synapseIndexMapDevice  GPU address of the EdgeIndexMap on device memory.
       virtual void advanceEdges(void* allSynapsesDevice, void* allNeuronsDevice, void* synapseIndexMapDevice) = 0;

       ///  Set some parameters used for advanceSynapsesDevice.
       virtual void setAdvanceSynapsesDeviceParams() = 0;

       ///  Set edge class ID defined by enumClassSynapses for the caller's Edge class.
       ///  The class ID will be set to classSynapses_d in device memory,
       ///  and the classSynapses_d will be referred to call a device function for the
       ///  particular edge class.
       ///  Because we cannot use virtual function (Polymorphism) in device functions,
       ///  we use this scheme.
       ///  Note: we used to use a function pointer; however, it caused the growth_cuda crash
       ///  (see issue#137).
       virtual void setSynapseClassID() = 0;

       ///  Prints GPU SynapsesProps data.
       ///
       ///  @param  allSynapsesDeviceProps   GPU address of the corresponding SynapsesDeviceProperties struct on device memory.
       virtual void printGPUSynapsesProps( void* allSynapsesDeviceProps ) const = 0;

#else // !defined(USE_GPU)
public:
   ///  Advance all the Synapses in the simulation.
   ///  Update the state of all edges for a time step.
   ///
   ///  @param  vertices   The Vertex list to search from.
   ///  @param  edgeIndexMap   Pointer to EdgeIndexMap structure.
   virtual void advanceEdges(IAllVertices *vertices, EdgeIndexMap *edgeIndexMap) = 0;

   ///  Advance one specific Edge.
   ///
   ///  @param  iEdg      Index of the Edge to connect to.
   ///  @param  vertices   The Vertex list to search from.
   virtual void advanceEdge(const BGSIZE iEdg, IAllVertices *vertices) = 0;

   ///  Remove a edge from the network.
   ///
   ///  @param  neuronIndex   Index of a vertex to remove from.
   ///  @param  iEdg          Index of a edge to remove.
   virtual void eraseEdge(const int neuronIndex, const BGSIZE iEdg) = 0;

#endif // defined(USE_GPU)
};
