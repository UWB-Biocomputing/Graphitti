/**
 * @file AllSpikingNeurons.cpp
 * 
 * @ingroup Simulator/Vertices
 *
 * @brief A container of all spiking neuron data
 */

#include "AllSpikingNeurons.h"
#include "AllSpikingSynapses.h"
#include "XmlRecorder.h"

///  Setup the internal structure of the class (allocate memories).
void AllSpikingNeurons::setupVertices()
{
   AllVertices::setupVertices();

   int maxSpikes = static_cast<int>(Simulator::getInstance().getEpochDuration()
                                    * Simulator::getInstance().getMaxFiringRate());

   hasFired_.assign(size_, false);
   vertexEvents_.assign(size_, maxSpikes);

   // register variable vertexEvents_
   Recorder &recorder = Simulator::getInstance().getModel().getRecorder();
   recorder.registerVariable("Neuron_", vertexEvents_);

   // // register neuron information in vertexEvents_ one by one
   // Recorder &recorder = Simulator::getInstance().getModel().getRecorder();
   // for (int iNeuron = 0; iNeuron < vertexEvents_.size(); iNeuron++) {
   //    string neuronID = "Neuron_" + to_string(iNeuron);
   //    recorder.registerVariable(neuronID, vertexEvents_[iNeuron]);
   // }
}

///  Clear the spike counts out of all Neurons.
void AllSpikingNeurons::clearSpikeCounts()
{
   // for (int i = 0; i < vertexEvents_.size(); i++) {
   //    vertexEvents_[i].clear();
   // }
   for (int i = 0; i < vertexEvents_.size(); i++) {
      vertexEvents_[i].startNewEpoch();
   }
}

#if !defined(USE_GPU)

///  Update internal state of the indexed Neuron (called by every simulation step).
///  Notify outgoing synapses if neuron has fired.
///
///  @param  synapses         The Synapse list to search from.
///  @param  edgeIndexMap  Reference to the EdgeIndexMap.
void AllSpikingNeurons::advanceVertices(AllEdges &synapses, const EdgeIndexMap &edgeIndexMap)
{
   int maxSpikes = (int)((Simulator::getInstance().getEpochDuration()
                          * Simulator::getInstance().getMaxFiringRate()));

   AllSpikingSynapses &spSynapses = dynamic_cast<AllSpikingSynapses &>(synapses);
   // For each neuron in the network
   for (int idx = Simulator::getInstance().getTotalVertices() - 1; idx >= 0; --idx) {
      // advance neurons
      advanceNeuron(idx);

      // notify outgoing/incoming synapses if neuron has fired
      if (hasFired_[idx]) {
         LOG4CPLUS_DEBUG(vertexLogger_,
                         "Neuron: " << idx << " has fired at time: "
                                    << g_simulationStep * Simulator::getInstance().getDeltaT());

         assert(vertexEvents_[idx].getNumEventsInEpoch() < maxSpikes);

         // notify outgoing synapses
         BGSIZE synapseCounts;


         synapseCounts = edgeIndexMap.outgoingEdgeCount_[idx];
         if (synapseCounts != 0) {
            int beginIndex = edgeIndexMap.outgoingEdgeBegin_[idx];
            BGSIZE iEdg;
            for (BGSIZE i = 0; i < synapseCounts; i++) {
               iEdg = edgeIndexMap.outgoingEdgeIndexMap_[beginIndex + i];
               spSynapses.preSpikeHit(iEdg);
            }
         }


         // notify incoming synapses
         synapseCounts = spSynapses.edgeCounts_[idx];
         BGSIZE synapse_notified = 0;

         if (spSynapses.allowBackPropagation()) {
            for (int z = 0; synapse_notified < synapseCounts; z++) {
               BGSIZE iEdg = Simulator::getInstance().getMaxEdgesPerVertex() * idx + z;
               if (spSynapses.inUse_[iEdg] == true) {
                  spSynapses.postSpikeHit(iEdg);
                  synapse_notified++;
               }
            }
         }

         hasFired_[idx] = false;
      }
   }
}

///  Fire the selected Neuron and calculate the result.
///
///  @param  index       Index of the Neuron to update.
void AllSpikingNeurons::fire(int index)
{
   // Note that the neuron has fired!
   hasFired_[index] = true;
   vertexEvents_[index].insertEvent(g_simulationStep);
}

///  Get the spike history of neuron[index] at the location offIndex.
///  More specifically, retrieves the global simulation time step for the spike
///  in question from the spike history record.
///
///  @param  index            Index of the neuron to get spike history.
///  @param  offIndex     Offset of the history buffer to get from. This indicates how many spikes
///                    in the past we are looking, so it must be a negative number (i.e., it is relative
///                    to the "current time", i.e., one location past the most recent spike). So, the
///                    most recent spike is offIndex = -1
uint64_t AllSpikingNeurons::getSpikeHistory(int index, int offIndex)
{
   // offIndex is a minus offset
   // This computes the index of a spike in the past (i.e., the most recent spike,
   // two spikes ago, etc). It starts with `spikeCountOffset_ + spikeCount_`,
   // which I believe at the end of an epoch should be one past the end of that
   // neuron's spikes in `spikeHistory_`. Then, the maximum number of spikes per
   // epoch is added. Then, the `offIndex` parameter is added. The expectation
   // is that `offIndex` will be a negative number (i.e., a spike in the past);
   // the reason that the max spikes value is added is to prevent this from
   // producing a negative total, so that finally taking mod max spikes will
   // "wrap around backwards" if needed.

   return vertexEvents_[index].getPastEvent(offIndex);
}

#endif
