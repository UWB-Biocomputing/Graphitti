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
   AsyncPhilox_d(int samplesPerGen, unsigned long seed);
   ~AsyncPhilox_d();
   void loadAsyncPhilox(int samplesPerSegment, unsigned long seed);
   void deleteDeviceStruct();
   float *requestSegment();

private:
   int numBlocks;
   int numThreads;
   int totalThreads;
   int segmentSize;
   int totalSegments;
   int bufferSize;
   unsigned long seed;

#ifdef ENABLE_NVTX
   int nvtxMarker;
   int nvtxCurrentMarker;
#endif


   cudaStream_t stream;

   float *buffers[2];
   int currentBuffer;
   int segmentIndex;

   curandStatePhilox4_32_10_t *spStates;

   // FILE* logfile;
   // float* hostBuffer;
   log4cplus::Logger
      consoleLogger_;   /// Logger for printing to the console as well as the logging file

   void fillBuffer(int bufferIndex);
};
