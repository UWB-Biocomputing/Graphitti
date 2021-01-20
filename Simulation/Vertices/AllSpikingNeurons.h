/**
 * @file AllSpikingNeurons.h
 * 
 * @ingroup Simulation/Vertices
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

#include "Global.h"
#include "AllNeurons.h"
#include "AllSpikingSynapses.h"

struct AllSpikingNeuronsDeviceProperties;

class AllSpikingNeurons : public AllNeurons {
public:
   AllSpikingNeurons();

   virtual ~AllSpikingNeurons();

   ///  Setup the internal structure of the class.
   ///  Allocate memories to store all neurons' state.
   virtual void setupNeurons();

   ///  Clear the spike counts out of all Neurons.
   void clearSpikeCounts();

#if defined(USE_GPU)
   public:
       ///  Set some parameters used for advanceNeuronsDevice.
       ///
       ///  @param  synapses               Reference to the allSynapses struct on host memory.
       virtual void setAdvanceNeuronsDeviceParams(IAllSynapses &synapses);

       ///  Copy spike counts data stored in device memory to host.
       ///
       ///  @param  allNeuronsDevice   GPU address of the allNeurons struct on device memory.
       virtual void copyNeuronDeviceSpikeCountsToHost( void* allNeuronsDevice) = 0;

       ///  Copy spike history data stored in device memory to host.
       ///
       ///  @param  allNeuronsDevice   GPU address of the allNeurons struct on device memory.
       virtual void copyNeuronDeviceSpikeHistoryToHost( void* allNeuronsDevice) = 0;

       ///  Clear the spike counts out of all neurons.
       ///
       ///  @param  allNeuronsDevice   GPU address of the allNeurons struct on device memory.
       virtual void clearNeuronSpikeCounts( void* allNeuronsDevice) = 0;

   protected:
       ///  Copy spike history data stored in device memory to host.
       ///  (Helper function of copyNeuronDeviceSpikeHistoryToHost)
       ///
       ///  @param  allNeuronsDevice   GPU address of the allNeurons struct on device memory.
       void copyDeviceSpikeHistoryToHost( AllSpikingNeuronsDeviceProperties& allNeuronsDevice);

       ///  Copy spike counts data stored in device memory to host.
       ///  (Helper function of copyNeuronDeviceSpikeCountsToHost)
       ///
       ///  @param  allNeuronsDevice   GPU address of the allNeurons struct on device memory.
       void copyDeviceSpikeCountsToHost( AllSpikingNeuronsDeviceProperties& allNeuronsDevice);

       ///  Clear the spike counts out of all neurons in device memory.
       ///  (helper function of clearNeuronSpikeCounts)
       ///
       ///  @param  allNeuronsDevice   GPU address of the allNeurons struct on device memory.
       void clearDeviceSpikeCounts( AllSpikingNeuronsDeviceProperties& allNeuronsDevice);
#else // !defined(USE_GPU)

public:
   ///  Update internal state of the indexed Neuron (called by every simulation step).
   ///  Notify outgoing synapses if neuron has fired.
   ///
   ///  @param  synapses         The Synapse list to search from.
   ///  @param  synapseIndexMap  Reference to the SynapseIndexMap.
   virtual void advanceNeurons(IAllSynapses &synapses, const SynapseIndexMap *synapseIndexMap);

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

#endif // defined(USE_GPU)

public:
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

protected:
   ///  True if back propagaion is allowed.
   ///  (parameters used for advanceNeuronsDevice.)
   bool fAllowBackPropagation_;

};

#if defined(USE_GPU)
struct AllSpikingNeuronsDeviceProperties : public AllNeuronsDeviceProperties
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
#endif // defined(USE_GPU)
