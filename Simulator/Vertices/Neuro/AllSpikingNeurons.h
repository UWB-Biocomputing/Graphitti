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


#include "AllSpikingSynapses.h"
#include "AllVertices.h"
#include "Global.h"

struct AllSpikingNeuronsDeviceProperties;

class AllSpikingNeurons : public AllVertices {
	public:
		AllSpikingNeurons();

		~AllSpikingNeurons() override;

		///  Setup the internal structure of the class.
		///  Allocate memories to store all neurons' state.
		void setupVertices() override;

		///  Clear the spike counts out of all Neurons.
		void clearSpikeCounts();

#ifdef __CUDACC__
   public:
       ///  Set some parameters used for advanceVerticesDevice.
       ///
       ///  @param  synapses               Reference to the allEdges struct on host memory.
       virtual void setAdvanceVerticesDeviceParams(AllEdges &synapses);

       ///  Copy spike counts data stored in device memory to host.
       ///
       ///  @param  allVerticesDevice   GPU address of the allVertices struct on device memory.
       virtual void copyNeuronDeviceSpikeCountsToHost( void* allVerticesDevice) = 0;

       ///  Copy spike history data stored in device memory to host.
       ///
       ///  @param  allVerticesDevice   GPU address of the allVertices struct on device memory.
       virtual void copyNeuronDeviceSpikeHistoryToHost( void* allVerticesDevice) = 0;

       ///  Clear the spike counts out of all neurons.
       ///
       ///  @param  allVerticesDevice   GPU address of the allVertices struct on device memory.
       virtual void clearNeuronSpikeCounts( void* allVerticesDevice) = 0;

   protected:
       ///  Copy spike history data stored in device memory to host.
       ///  (Helper function of copyNeuronDeviceSpikeHistoryToHost)
       ///
       ///  @param  allVerticesDevice   GPU address of the allVertices struct on device memory.
       void copyDeviceSpikeHistoryToHost( AllSpikingNeuronsDeviceProperties& allVerticesDevice);

       ///  Copy spike counts data stored in device memory to host.
       ///  (Helper function of copyNeuronDeviceSpikeCountsToHost)
       ///
       ///  @param  allVerticesDevice   GPU address of the allVertices struct on device memory.
       void copyDeviceSpikeCountsToHost( AllSpikingNeuronsDeviceProperties& allVerticesDevice);

       ///  Clear the spike counts out of all neurons in device memory.
       ///  (helper function of clearNeuronSpikeCounts)
       ///
       ///  @param  allVerticesDevice   GPU address of the allVertices struct on device memory.
       void clearDeviceSpikeCounts( AllSpikingNeuronsDeviceProperties& allVerticesDevice);
#else // !defined(__CUDACC__)

	public:
		///  Update internal state of the indexed Neuron (called by every simulation step).
		///  Notify outgoing synapses if neuron has fired.
		///
		///  @param  synapses         The Synapse list to search from.
		///  @param  edgeIndexMap  Reference to the EdgeIndexMap.
		void advanceVertices(AllEdges& synapses, const EdgeIndexMap* edgeIndexMap) override;

		/// Get the spike history of neuron[index] at the location offIndex.
		/// More specifically, retrieves the global simulation time step for the spike
		/// in question from the spike history record.
		///
		/// @param  index            Index of the neuron to get spike history.
		/// @param  offIndex         Offset of the history buffer to get from.
		uint64_t getSpikeHistory(int index, int offIndex);

	protected:
		///  Helper for #advanceNeuron. Updates state of a single neuron.
		///
		///  @param  index            Index of the neuron to update.
		virtual void advanceNeuron(const int index) = 0;

		///  Initiates a firing of a neuron to connected neurons
		///
		///  @param  index            Index of the neuron to fire.
		virtual void fire(const int index) const;

#endif // defined(__CUDACC__)

	public:
		///  The booleans which track whether the neuron has fired.
		bool* hasFired_;

		///  The number of spikes since the last growth cycle.
		int* spikeCount_;

		///  Offset of the spike_history buffer.
		int* spikeCountOffset_;

		///  Step count (history) for each spike fired by each neuron.
		///  The step counts are stored in a buffer for each neuron, and the pointers
		///  to the buffer are stored in a list pointed by spike_history.
		///  Each buffer is a circular, and offset of top location of the buffer i is
		///  specified by spikeCountOffset[i].
		uint64_t** spikeHistory_;

	protected:
		///  True if back propagaion is allowed.
		///  (parameters used for advanceVerticesDevice.)
		bool fAllowBackPropagation_;

};

#ifdef __CUDACC__
struct AllSpikingNeuronsDeviceProperties : public AllVerticesDeviceProperties
{
        ///  The booleans which track whether the neuron has fired.
        bool *hasFired_;

        ///  The number of spikes since the last growth cycle.
        int *spikeCount_;

        ///  Offset of the spike_history buffer.
        int *spikeCountOffset_;

        ///  Step count (history) for each spike fired by each neuron.
        ///  The step counts are stored in a buffer for each neuron, and the pointers
        ///  to the buffer are stored in a list pointed by spike_history. 
        ///  Each buffer is a circular, and offset of top location of the buffer i is
        ///  specified by spikeCountOffset[i].
        uint64_t **spikeHistory_;
};
#endif // defined(__CUDACC__)
