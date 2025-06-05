#pragma once
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <log4cplus/loggingmacros.h>
#include "Book.h"
#include <cstdio>
#include <cstdlib>
class AsyncMT_d {
public:
   AsyncMT_d() = default;
   AsyncMT_d(int samplesPerGen, unsigned long seed);
   ~AsyncMT_d();
   void loadAsyncMT(int samplesPerSegment, unsigned long seed);
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

   curandStatePhilox4_32_10_t* spStates;

   // FILE* logfile;
   // float* hostBuffer;
   log4cplus::Logger
      consoleLogger_;   /// Logger for printing to the console as well as the logging file

   void fillBuffer(int bufferIndex);
};
