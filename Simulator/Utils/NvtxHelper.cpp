/**
 * @file NvtxHelper.cpp
 * 
 * @ingroup Simulator/Utils
 * 
 * @brief Helper functions to enable nvtx profiling
 * When ENABLE_NVTX is false the functions are replaced with blank inline functions which are removed by the compiler
 * This file is only included in the utils library when ENABLE_CUDA=YES
 */

#include "NvtxHelper.h"
#include <cuda_runtime.h>
#include <nvToolsExt.h>

void nvtxPushColor(const std::string &name, Color pColor)
{
   nvtxEventAttributes_t eventAttrib = {};
   eventAttrib.version = NVTX_VERSION;
   eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
   eventAttrib.colorType = NVTX_COLOR_ARGB;
   eventAttrib.color = static_cast<uint32_t>(pColor);
   eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
   eventAttrib.message.ascii = name.c_str();

   nvtxRangePushEx(&eventAttrib);
}

void nvtxPop()
{
   nvtxRangePop();
}