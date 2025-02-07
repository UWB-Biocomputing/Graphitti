#include "NvtxHelper.h"
#include <cuda_runtime.h>
#include <nvToolsExt.h>

void nvtxPushColor(const std::string &name, uint32_t color)
{
   nvtxEventAttributes_t eventAttrib = {};
   eventAttrib.version = NVTX_VERSION;
   eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
   eventAttrib.colorType = NVTX_COLOR_ARGB;
   eventAttrib.color = color;
   eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
   eventAttrib.message.ascii = name.c_str();

   nvtxRangePushEx(&eventAttrib);
}

void nvtxPop()
{
   nvtxRangePop();
}