#ifndef NVTX_HELPER_H
#define NVTX_HELPER_H

#include <cstdint>
#include <string>

// Define NVTX colors (ARGB format)
#define RED 0xFFFF0000      // Red
#define GREEN 0xFF00FF00    // Green
#define BLUE 0xFF0000FF     // Blue
#define YELLOW 0xFFFFFF00   // Yellow
#define ORANGE 0xFFFFA500   // Orange
#define PURPLE 0xFF800080   // Purple

#ifdef ENABLE_NVTX

// Function to push an NVTX range with a given name and color
void nvtxPushColor(const std::string &name, uint32_t color);

// Function to pop the most recent NVTX range
void nvtxPop();

#else
inline void nvtxPushColor(const std::string &, uint32_t)
{
}
inline void nvtxPop()
{
}

#endif   // ENABLE_NVTX


#endif   // NVTX_HELPER_H