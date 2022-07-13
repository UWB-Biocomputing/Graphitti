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
extern CUDA_CALLABLE void preSpikingSynapsesSpikeHitDevice( const BGSIZE iEdg, AllSpikingSynapsesDeviceProperties* allEdgesDevice );
extern CUDA_CALLABLE void postSpikingSynapsesSpikeHitDevice( const BGSIZE iEdg, AllSpikingSynapsesDeviceProperties* allEdgesDevice );
extern CUDA_CALLABLE void postSTDPSynapseSpikeHitDevice( const BGSIZE iEdg, AllSTDPSynapsesDeviceProperties* allEdgesDevice );


#endif
