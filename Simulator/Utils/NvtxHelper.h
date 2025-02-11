/**
 * @file NvtxHelper.h
 * 
 * @ingroup Simulator/Utils
 * 
 * @brief Helper functions to enable nvtx profiling
 * When ENABLE_NVTX is false the functions are replaced with blank inline functions which are removed by the compiler
 */

#ifndef NVTX_HELPER_H
#define NVTX_HELPER_H

#include <cstdint>
#include <string>

// Define NVTX colors (ARGB format)
enum class Color : std::uint32_t {
   RED = 0xFFFF0000,
   GREEN = 0xFF00FF00,
   BLUE = 0xFF0000FF,
   YELLOW = 0xFFFFFF00,
   ORANGE = 0xFFFFA500,
   PURPLE = 0xFF800080
};

#ifdef ENABLE_NVTX

// Function to push an NVTX range with a given name and color
void nvtxPushColor(const std::string &name, Color pColor);

// Function to pop the most recent NVTX range
void nvtxPop();

#else
inline void nvtxPushColor(const std::string &, Color)
{
}
inline void nvtxPop()
{
}

#endif   // ENABLE_NVTX


#endif   // NVTX_HELPER_H