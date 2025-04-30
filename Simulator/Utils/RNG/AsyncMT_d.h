#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32_kernel.h>
#include <curand_mtgp32dc_p_11213.h>   // Precomputed parameter table

class AsyncMT_d {
public:
   AsyncMT_d(int samplesPerGen, unsigned long seed);
   ~AsyncMT_d();

   float *requestSegment();

private:
   int numGenerators;
   int segmentSize;
   int totalSegments;
   int bufferSize;
   int totalSamples;
   unsigned long seed;

   cudaStream_t stream;

   float *buffers[2];
   int currentBuffer;
   int segmentIndex;

   curandStateMtgp32 *d_states;
   mtgp32_kernel_params_t *d_params;

   void fillBuffer(int bufferIndex);
};
