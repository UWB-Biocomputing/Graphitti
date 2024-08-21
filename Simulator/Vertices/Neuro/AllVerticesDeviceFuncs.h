/**
 * @file AllVerticesDeviceFuncs.h
 * 
 * @ingroup Simulator/Vertices
 *
 * @brief Device functions for vertices
 */

#pragma once

#include "AllIZHNeurons.h"
#include "AllSTDPSynapses.h"
#include "AllSynapsesDeviceFuncs.h"


#if defined(__CUDACC__)
extern CUDA_CALLABLE void
   preSpikingSynapsesSpikeHitDevice(BGSIZE iEdg,
                                    AllSpikingSynapsesDeviceProperties *allEdgesDevice);
extern CUDA_CALLABLE void
   postSpikingSynapsesSpikeHitDevice(BGSIZE iEdg,
                                     AllSpikingSynapsesDeviceProperties *allEdgesDevice);
extern CUDA_CALLABLE void
   postSTDPSynapseSpikeHitDevice(BGSIZE iEdg, AllSTDPSynapsesDeviceProperties *allEdgesDevice);


#endif
