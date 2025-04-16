/**
 *  @file AllSpikingSynapses.h
 *
 *  @ingroup Simulator/Edges
 *
 *  @brief A container of all spiking synapse data
 *
 *  The container holds synapse parameters of all synapses.
 *  Each kind of synapse parameter is stored in a 2D array. Each item in the first
 *  dimension of the array corresponds with each neuron, and each item in the second
 *  dimension of the array corresponds with a synapse parameter of each synapse of the neuron.
 * Because each neuron owns different number of synapses, the number of synapses
 *  for each neuron is stored in a 1D array, edge_counts.
 *
 *  For CUDA implementation, we used another structure, AllEdgesDevice, where synapse
 *  parameters are stored in 1D arrays instead of 2D arrays, so that device functions
 *  can access these data with less latency. When copying a synapse parameter, P[i][j],
 *  from host to device, it is stored in P[i * max_edges_per_vertex + j] in
 *  AllEdgesDevice structure.
 *
 *  The latest implementation uses the identical data structure between host and CUDA;
 *  that is, synapse parameters are stored in a 1D array, so we don't need conversion
 *  when copying data between host and device memory.
 */
#pragma once

#include "AllNeuroEdges.h"
// cereal
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>

struct AllSpikingSynapsesDeviceProperties;

using fpPreSynapsesSpikeHit_t = void (*)(const BGSIZE, AllSpikingSynapsesDeviceProperties *);

using fpPostSynapsesSpikeHit_t = void (*)(const BGSIZE, AllSpikingSynapsesDeviceProperties *);


class AllSpikingSynapses : public AllNeuroEdges {
public:
   AllSpikingSynapses();

   AllSpikingSynapses(int numVertices, int maxEdges);

   virtual ~AllSpikingSynapses() = default;

   static AllEdges *Create()
   {
      return new AllSpikingSynapses();
   }

   ///  Setup the internal structure of the class (allocate memories and initialize them).
   virtual void setupEdges() override;

   ///  Reset time varying state vars and recompute decay.
   ///
   ///  @param  iEdg     Index of the synapse to set.
   ///  @param  deltaT   Inner simulation step duration
   virtual void resetEdge(BGSIZE iEdg, BGFLOAT deltaT) override;

   /// Load member variables from configuration file. Registered to OperationManager as Operation::op::loadParameters
   virtual void loadParameters() override;

   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const override;

   ///  Create a Synapse and connect it to the model.
   ///
   ///  @param  iEdg        Index of the synapse to set.
   ///  @param  srcVertex   Index of the source Neuron.
   ///  @param  destVertex  Index of the destination Neuron.
   ///  @param  deltaT      Inner simulation step duration.
   ///  @param  type        Type of the Synapse to create.
   virtual void createEdge(BGSIZE iEdg, int srcVertex, int destVertex, BGFLOAT deltaT,
                           edgeType type) override;

   ///  Check if the back propagation (notify a spike event to the pre neuron)
   ///  is allowed in the synapse class.
   ///
   ///  @return true if the back propagation is allowed.
   virtual bool allowBackPropagation();

   ///  Prints SynapsesProps data to console.
   virtual void printSynapsesProps() const;

   ///  Cereal serialization method
   template <class Archive> void serialize(Archive &archive);

protected:
   ///  Setup the internal structure of the class (allocate memories and initialize them).
   ///
   ///  @param  numVertices   Total number of vertices in the network.
   ///  @param  maxEdges  Maximum number of synapses per neuron.
   virtual void setupEdges(int numVertices, int maxEdges);

   ///  Initializes the queues for the Synapse.
   ///
   ///  @param  iEdg   index of the synapse to set.
   virtual void initSpikeQueue(BGSIZE iEdg);

   ///  Updates the decay if the synapse selected.
   ///
   ///  @param  iEdg    Index of the synapse to set.
   ///  @param  deltaT  Inner simulation step duration
   ///  @return true is success.
   bool updateDecay(BGSIZE iEdg, BGFLOAT deltaT);

   ///  Sets the data for Synapse to input's data.
   ///
   ///  @param  input  istream to read from.
   ///  @param  iEdg   Index of the synapse to set.
   virtual void readEdge(istream &input, BGSIZE iEdg) override;

   ///  Write the synapse data to the stream.
   ///
   ///  @param  output  stream to print out to.
   ///  @param  iEdg    Index of the synapse to print out.
   virtual void writeEdge(ostream &output, BGSIZE iEdg) const override;

#if defined(USE_GPU)
public:
   ///  Allocate GPU memories to store all synapses' states,
   ///  and copy them from host to GPU memory.
   virtual void allocEdgeDeviceStruct() override;

   ///  Allocate GPU memories to store all synapses' states,
   ///  and copy them from host to GPU memory.
   ///
   ///  @param  allEdgesDevice     GPU address of the allEdges struct on device memory.
   ///  @param  numVertices            Number of vertices.
   ///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
   virtual void allocEdgeDeviceStruct(void **allEdgesDevice, int numVertices,
                                      int maxEdgesPerVertex) override;

   ///  Delete GPU memories.
   ///
   ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
   virtual void deleteEdgeDeviceStruct(void *allEdgesDevice) override;

   ///  Copy all synapses' data from host to device.
   ///
   virtual void copyEdgeHostToDevice() override;

   ///  Copy all synapses' data from host to device.
   ///
   ///  @param  allEdgesDevice     GPU address of the allEdges struct on device memory.
   ///  @param  numVertices            Number of vertices.
   ///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
   virtual void copyEdgeHostToDevice(void *allEdgesDevice, int numVertices,
                                     int maxEdgesPerVertex) override;

   /// Copy all synapses' data from device to host.
   ///
   ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
   virtual void copyEdgeDeviceToHost(void *allEdgesDevice) override;

   ///  Get edge_counts in AllNeuroEdges struct on device memory.
   ///
   ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
   virtual void copyDeviceEdgeCountsToHost(void *allEdgesDevice) override;

   ///  Get summationCoord and in_use in AllNeuroEdges struct on device memory.
   ///
   ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
   virtual void copyDeviceEdgeSumIdxToHost(void *allEdgesDevice) override;

   ///  Advance all the Synapses in the simulation.
   ///  Update the state of all synapses for a time step.
   ///
   ///  @param  allEdgesDevice      GPU address of the allEdges struct on device memory.
   ///  @param  allVerticesDevice       GPU address of the allNeurons struct on device memory.
   ///  @param  edgeIndexMapDevice  GPU address of the EdgeIndexMap on device memory.
   virtual void advanceEdges(void *allEdgesDevice, void *allVerticesDevice,
                             void *edgeIndexMapDevice) override;

   ///  Set some parameters used for advanceEdgesDevice.
   ///  Currently we set a member variable: m_fpChangePSR_h.
   virtual void setAdvanceEdgesDeviceParams() override;

   ///  Set synapse class ID defined by enumClassSynapses for the caller's Synapse class.
   ///  The class ID will be set to classSynapses_d in device memory,
   ///  and the classSynapses_d will be referred to call a device function for the
   ///  particular synapse class.
   ///  Because we cannot use virtual function (Polymorphism) in device functions,
   ///  we use this scheme.
   ///  Note: we used to use a function pointer; however, it caused the growth_cuda crash
   ///  (see issue#137).
   virtual void setEdgeClassID() override;

   ///  Prints GPU SynapsesProps data.
   ///
   ///  @param  allEdgesDeviceProps   GPU address of the corresponding SynapsesDeviceProperties struct on device memory.
   virtual void printGPUEdgesProps(void *allEdgesDeviceProps) const override;

protected:
   ///  Allocate GPU memories to store all synapses' states,
   ///  and copy them from host to GPU memory.
   ///  (Helper function of allocEdgeDeviceStruct)
   ///
   ///  @param  allEdgesDevice     GPU address of the allEdges struct on device memory.
   ///  @param  numVertices            Number of vertices.
   ///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
   void allocDeviceStruct(AllSpikingSynapsesDeviceProperties &allEdgesDevice, int numVertices,
                          int maxEdgesPerVertex);

   ///  Delete GPU memories.
   ///  (Helper function of deleteEdgeDeviceStruct)
   ///
   ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
   void deleteDeviceStruct(AllSpikingSynapsesDeviceProperties &allEdgesDevice);

   ///  Copy all synapses' data from host to device.
   ///  (Helper function of copyEdgeHostToDevice)
   ///
   ///  @param  allEdgesDevice     GPU address of the allEdges struct on device memory.
   ///  @param  numVertices            Number of vertices.
   ///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
   void copyHostToDevice(void *allEdgesDevice,
                         AllSpikingSynapsesDeviceProperties &allEdgesDeviceProps, int numVertices,
                         int maxEdgesPerVertex);

   ///  Copy all synapses' data from device to host.
   ///  (Helper function of copyEdgeDeviceToHost)
   ///
   ///  @param  allEdgesDevice     GPU address of the allEdges struct on device memory.
   ///  @param  numVertices            Number of vertices.
   ///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
   void copyDeviceToHost(AllSpikingSynapsesDeviceProperties &allEdgesDevice);
#else   // !defined(USE_GPU)
public:
   ///  Advance one specific Synapse.
   ///
   ///  @param  iEdg      Index of the Synapse to connect to.
   ///  @param  neurons   The Neuron list to search from.
   virtual void advanceEdge(BGSIZE iEdg, AllVertices &neurons) override;

   ///  Prepares Synapse for a spike hit.
   ///
   ///  @param  iEdg   Index of the Synapse to update.
   virtual void preSpikeHit(BGSIZE iEdg);

   ///  Prepares Synapse for a spike hit (for back propagation).
   ///
   ///  @param  iEdg   Index of the Synapse to update.
   virtual void postSpikeHit(BGSIZE iEdg);

protected:
   ///  Checks if there is an input spike in the queue.
   ///
   ///  @param  iEdg   Index of the Synapse to connect to.
   ///  @return true if there is an input spike event.
   bool isSpikeQueue(BGSIZE iEdg);

   ///  Calculate the post synapse response after a spike.
   ///
   ///  @param  iEdg        Index of the synapse to set.
   ///  @param  deltaT      Inner simulation step duration.
   virtual void changePSR(BGSIZE iEdg, BGFLOAT deltaT);

#endif

public:
   ///  The decay for the psr.
   vector<BGFLOAT> decay_;

   ///  The synaptic time constant \f$\tau\f$ [units=sec; range=(0,100)].
   vector<BGFLOAT> tau_;

   BGFLOAT tau_II_;
   BGFLOAT tau_IE_;
   BGFLOAT tau_EI_;
   BGFLOAT tau_EE_;
   BGFLOAT delay_II_;
   BGFLOAT delay_IE_;
   BGFLOAT delay_EI_;
   BGFLOAT delay_EE_;

#define BYTES_OF_DELAYQUEUE (sizeof(uint32_t) / sizeof(uint8_t))
#define LENGTH_OF_DELAYQUEUE (BYTES_OF_DELAYQUEUE * 8)

   ///  The synaptic transmission delay, descretized into time steps.
   vector<int> totalDelay_;

   ///  Pointer to the delayed queue.
   vector<uint32_t> delayQueue_;

   ///  The index indicating the current time slot in the delayed queue
   ///  Note: This variable is used in GpuSim_struct.cu but I am not sure
   ///  if it is actually from a synapse. Will need a little help here. -Aaron
   ///  Note: This variable can be GLOBAL VARIABLE, but need to modify the code.
   vector<int> delayIndex_;

   ///  Length of the delayed queue.
   vector<int> delayQueueLength_;
};

#if defined(USE_GPU)

CUDA_CALLABLE bool
   isSpikingSynapsesSpikeQueueDevice(AllSpikingSynapsesDeviceProperties *allEdgesDevice,
                                     BGSIZE iEdg);
CUDA_CALLABLE void
   changeSpikingSynapsesPSRDevice(AllSpikingSynapsesDeviceProperties *allEdgesDevice, BGSIZE iEdg,
                                  const uint64_t simulationStep, BGFLOAT deltaT);

struct AllSpikingSynapsesDeviceProperties : public AllEdgesDeviceProperties {
   ///  The decay for the psr.
   BGFLOAT *decay_;

   ///  The synaptic time constant \f$\tau\f$ [units=sec; range=(0,100)].
   BGFLOAT *tau_;

   ///  The synaptic transmission delay, descretized into time steps.
   int *totalDelay_;

   ///  Pointer to the delayed queue.
   uint32_t *delayQueue_;

   ///  The index indicating the current time slot in the delayed queue
   ///  Note: This variable is used in GpuSim_struct.cu but I am not sure
   ///  if it is actually from a synapse. Will need a little help here. -Aaron
   ///  Note: This variable can be GLOBAL VARIABLE, but need to modify the code.
   int *delayIndex_;

   ///  Length of the delayed queue.
   int *delayQueueLength_;
};
#endif   // defined(USE_GPU)

CEREAL_REGISTER_TYPE(AllSpikingSynapses);

///  Cereal serialization method
template <class Archive> void AllSpikingSynapses::serialize(Archive &archive)
{
   archive(cereal::base_class<AllNeuroEdges>(this), cereal::make_nvp("decay", decay_),
           cereal::make_nvp("tau", tau_), cereal::make_nvp("tau_II", tau_II_),
           cereal::make_nvp("tau_IE", tau_IE_), cereal::make_nvp("tau_EI", tau_EI_),
           cereal::make_nvp("tau_EE", tau_EE_), cereal::make_nvp("delay_II", delay_II_),
           cereal::make_nvp("delay_IE", delay_IE_), cereal::make_nvp("delay_EI", delay_EI_),
           cereal::make_nvp("delay_EE", delay_EE_), cereal::make_nvp("totalDelay", totalDelay_),
           cereal::make_nvp("delayQueue", delayQueue_), cereal::make_nvp("delayIndex", delayIndex_),
           cereal::make_nvp("delayQueueLength", delayQueueLength_));
}
