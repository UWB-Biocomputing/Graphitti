/**
 * @file AsyncPhilox_d.h
 * 
 * @ingroup Simulator/Utils/RNG
 * 
 * @brief Asynchronous Philox RNG using curand to fill GPU buffers
 * 
 * AsyncPhilox_d class maintains two large GPU buffers for noise.
 * GPUModel calls loadAsyncPhilox to initialize states and
 * fill the buffers, then, each advance requestSegment
 * returns a float* slice of a buffer for use in
 * advanceVertices
 */

#pragma once
#include "Book.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <log4cplus/loggingmacros.h>
class AsyncPhilox_d {
public:
   AsyncPhilox_d() = default;

   ~AsyncPhilox_d();

   /// Initializes generator and allocates device memory
   /// @param samplesPerSegment Number of total vertices
   /// @param seed RNG seed.
   void loadAsyncPhilox(int samplesPerSegment, unsigned long seed);

   /// Free device memory
   void deleteDeviceStruct();

   /// Request a new segment of generated noise.
   /// @return Pointer to a slice of device memory containing noise.
   float *requestSegment();

private:
   /// Number of CUDA blocks to launch per kernel call.
   int numBlocks_;

   /// Number of threads per CUDA block.
   int numThreads_;

   /// Total number of threads = numBlocks × numThreads.
   int totalThreads_;

   /// Number of random floats per segment.
   int segmentSize_;

   /// Number of total segments in each buffer.
   int totalSegments_;

   /// Number of random floats per buffer.
   int bufferSize_;

   /// RNG seed.
   unsigned long seed_;

#ifdef ENABLE_NVTX
   /// Marker index for NVTX profiling (if enabled).
   int nvtxMarker_;

   /// Tracks current NVTX marker for alternating regions.
   int nvtxCurrentMarker_;
#endif

   /// CUDA stream used for asynchronous kernel launches.
   cudaStream_t RNG_stream_;

   /// Double-buffered random number output on device.
   float *buffers_d[2];

   /// Index of currently active buffer.
   int currentBuffer_;

   /// Index of the next segment to serve.
   int segmentIndex_;

   /// Device-side array of Philox curand RNG states.
   curandStatePhilox4_32_10_t *spStates_d;

   // FILE* logfile;
   // float* hostBuffer;
   /// Logger for printing to the console as well as the logging file
   log4cplus::Logger
      consoleLogger_;   

   /// Internal helper to fill a specified buffer with random floats.
   /// @param bufferIndex Index (0 or 1) of the buffer to fill.
   void fillBuffer(int bufferIndex);
};
