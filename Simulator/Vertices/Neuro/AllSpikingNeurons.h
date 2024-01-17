/**
 * @file AllSpikingNeurons.h
 * 
 * @ingroup Simulator/Vertices
 *
 * @brief A container of all spiking neuron data
 * 
 * This is the base class of all spiking neuron classes.
 *
 * The class uses a data-centric structure, which utilizes a structure as the containers of
 * all neuron.
 *
 * The container holds neuron parameters of all neurons.
 * Each kind of neuron parameter is stored in a 1D array, of which length
 * is number of all neurons. Each array of a neuron parameter is pointed by a
 * corresponding member variable of the neuron parameter in the class.
 * 
 * This structure was originally designed for the GPU implementation of the
 * simulator, and this refactored version of the simulator simply uses that design for
 * all other implementations as well. This is to simplify transitioning from
 * single-threaded to multi-threaded.
 */

#pragma once
using namespace std;
#include "AllSpikingSynapses.h"
#include "AllVertices.h"
#include "EventBuffer.h"
#include "Global.h"
#include <vector>
// cereal
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>

struct AllSpikingNeuronsDeviceProperties;

class AllSpikingNeurons : public AllVertices {
public:
   AllSpikingNeurons() = default;

   virtual ~AllSpikingNeurons() = default;

   ///  Setup the internal structure of the class.
   ///  Allocate memories to store all neurons' state.
   virtual void setupVertices() override;

   ///  Clear the spike counts out of all Neurons.
   void clearSpikeCounts();

   ///  Cereal serialization method
   template <class Archive> void serialize(Archive &archive);

#if defined(USE_GPU)
public:
   ///  Set some parameters used for advanceVerticesDevice.
   ///
   ///  @param  synapses               Reference to the allEdges struct on host memory.
   virtual void setAdvanceVerticesDeviceParams(AllEdges &synapses);

   ///  Clear the spike counts out of all neurons.
   //
   ///  @param  allVerticesDevice   GPU address of the allVertices struct on device memory.
   virtual void clearNeuronSpikeCounts(void *allVerticesDevice) = 0;
   virtual void copyFromDevice(void *deviceAddress) override;
   virtual void copyToDevice(void *deviceAddress) override;

protected:
   ///  Clear the spike counts out of all neurons in device memory.
   ///  (helper function of clearNeuronSpikeCounts)
   ///
   ///  @param  allVerticesDevice   GPU address of the allVertices struct on device memory.
   void clearDeviceSpikeCounts(AllSpikingNeuronsDeviceProperties &allVerticesDevice);
#else   // !defined(USE_GPU)
public:
   ///  Update internal state of the indexed Neuron (called by every simulation step).
   ///  Notify outgoing synapses if neuron has fired.
   ///
   ///  @param  synapses         The Synapse list to search from.
   ///  @param  edgeIndexMap  Reference to the EdgeIndexMap.
   virtual void advanceVertices(AllEdges &synapses, const EdgeIndexMap &edgeIndexMap);

   /// Get the spike history of neuron[index] at the location offIndex.
   /// More specifically, retrieves the global simulation time step for the spike
   /// in question from the spike history record.
   ///
   /// @param  index            Index of the neuron to get spike history.
   /// @param  offIndex         Offset of the history buffer to get from.
   uint64_t getSpikeHistory(int index, int offIndex);

protected:
   /// helper for recorder register variables in setupVertices()
   /// Register spike history variables for all neurons.
   /// Option 1: Register neuron information in vertexEvents_ one by one.
   /// Option 2: Register a vector of EventBuffer variables.
   void registerSpikeHistoryVariables();

   ///  Helper for #advanceNeuron. Updates state of a single neuron.
   ///
   ///  @param  index            Index of the neuron to update.
   virtual void advanceNeuron(int index) = 0;

   ///  Initiates a firing of a neuron to connected neurons
   ///
   ///  @param  index            Index of the neuron to fire.
   virtual void fire(int index);

#endif   // defined(USE_GPU)

   // TODO change the "public" after regineering Recorder
public:
   ///  The booleans which track whether the neuron has fired.
   vector<bool> hasFired_;

   /// Holds at least one epoch's worth of event times for every vertex
   vector<EventBuffer> vertexEvents_;

protected:
   ///  True if back propagaion is allowed.
   ///  (parameters used for advanceVerticesDevice.)
   bool fAllowBackPropagation_;
};

// TODO: move this into EventBuffer.h. Well, hasFired_ and inherited members have to stay somehow.
#if defined(USE_GPU)
struct AllSpikingNeuronsDeviceProperties : public AllVerticesDeviceProperties {
   ///  The booleans which track whether the neuron has fired.
   bool *hasFired_;

   ///  Step count (history) for each spike fired by each neuron.
   ///  The step counts are stored in a buffer for each neuron, and the pointers
   ///  to the buffer are stored in a list pointed by spike_history.
   ///  Each buffer is a circular, and offset of top location of the buffer i is
   ///  specified by spikeCountOffset[i].
   uint64_t **spikeHistory_;
   int *queueFront_;
   int *queueEnd_;
   int *epochStart_;
   int *numEventsInEpoch_;
};
#endif   // defined(USE_GPU)

CEREAL_REGISTER_TYPE(AllSpikingNeurons);

///  Cereal serialization method
template <class Archive> void AllSpikingNeurons::serialize(Archive &archive)
{
   archive(cereal::base_class<AllVertices>(this), cereal::make_nvp("hasFired_", hasFired_),
           //   cereal::make_nvp("vertexEvents_", vertexEvents_),
           cereal::make_nvp("fAllowBackPropagation_", fAllowBackPropagation_));
}
