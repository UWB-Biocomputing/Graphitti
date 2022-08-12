/**
 * @file AllVerticesDeviceFuncs_d.cpp
 *
 * @ingroup Simulator/Vertices
 *
 * @brief Device functions for vertices
 */

// #include "AllSynapsesDeviceFuncs.h"
// #include "AllVerticesDeviceFuncs.h"

/******************************************
 * Device Functions for advanceVertices
 ******************************************/
///@{

// ///  Prepares Synapse for a spike hit.
// ///
// ///  @param[in] iEdg                  Index of the Synapse to update.
// ///  @param[in] allEdgesDevice     Pointer to AllSpikingSynapsesDeviceProperties
// ///  structures
// ///                                   on device memory.
// CUDA_CALLABLE void preSpikingSynapsesSpikeHitDevice(
//     const BGSIZE iEdg, AllSpikingSynapsesDeviceProperties *allEdgesDevice) {
//   uint32_t &delay_queue = allEdgesDevice->delayQueue_[iEdg];
//   int delayIdx = allEdgesDevice->delayIndex_[iEdg];
//   int ldelayQueue = allEdgesDevice->delayQueueLength_[iEdg];
//   int total_delay = allEdgesDevice->totalDelay_[iEdg];

//   // Add to spike queue

//   // calculate index where to insert the spike into delayQueue
//   int idx = delayIdx + total_delay;
//   if (idx >= ldelayQueue) {
//     idx -= ldelayQueue;
//   }

//   // set a spike
//   // assert( !(delay_queue[0] & (0x1 << idx)) );
//   delay_queue |= (0x1 << idx);
// }

// ///  Prepares Synapse for a spike hit (for back propagation).
// ///
// ///  @param[in] iEdg                  Index of the Synapse to update.
// ///  @param[in] allEdgesDevice     Pointer to AllSpikingSynapsesDeviceProperties
// ///  structures
// ///                                   on device memory.
// CUDA_CALLABLE void postSpikingSynapsesSpikeHitDevice(
//     const BGSIZE iEdg, AllSpikingSynapsesDeviceProperties *allEdgesDevice) {}

// ///  Prepares Synapse for a spike hit (for back propagation).
// ///
// ///  @param[in] iEdg                  Index of the Synapse to update.
// ///  @param[in] allEdgesDevice     Pointer to AllSTDPSynapsesDeviceProperties
// ///  structures
// ///                                   on device memory.
// CUDA_CALLABLE void
// postSTDPSynapseSpikeHitDevice(const BGSIZE iEdg,
//                               AllSTDPSynapsesDeviceProperties *allEdgesDevice) {
//   uint32_t &delayQueue = allEdgesDevice->delayQueuePost_[iEdg];
//   int delayIndex = allEdgesDevice->delayIndexPost_[iEdg];
//   int delayQueueLength = allEdgesDevice->delayQueuePostLength_[iEdg];
//   int totalDelay = allEdgesDevice->totalDelayPost_[iEdg];

//   // Add to spike queue

//   // calculate index where to insert the spike into delayQueue
//   int idx = delayIndex + totalDelay;
//   if (idx >= delayQueueLength) {
//     idx -= delayQueueLength;
//   }

//   // set a spike
//   // assert( !(delay_queue[0] & (0x1 << idx)) );
//   delayQueue |= (0x1 << idx);
// }
