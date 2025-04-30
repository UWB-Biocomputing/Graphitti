#include "AsyncMT_d.h"
#include <cassert>
#include <curand_mtgp32dc_p_11213.h>

__global__ void generateKernel(curandStateMtgp32 *state, float *output, int samplesPerGen)
{
   int tid = threadIdx.x;
   int gen_id = blockIdx.x;
   if (gen_id >= gridDim.x)
      return;

   curandStateMtgp32 localState = state[gen_id];
   for (int i = tid; i < samplesPerGen; i += blockDim.x) {
      output[gen_id * samplesPerGen + i] = curand_normal(&localState);
   }
   state[gen_id] = localState;
}

AsyncMT_d::AsyncMT_d(int samplesPerSegment, unsigned long seed) :
   segmentSize(samplesPerSegment), seed(seed), currentBuffer(0), segmentIndex(0)
{
   totalSegments = 10;   // Each buffer has 10 segments
   bufferSize = segmentSize * totalSegments;
   totalSamples = bufferSize * 2;
   numGenerators = 50;   //placeholder num of blocks

   // Create internal stream
   cudaStreamCreate(&stream);

   // Allocate two large buffers
   cudaMalloc(&buffers[0], bufferSize * sizeof(float));
   cudaMalloc(&buffers[1], bufferSize * sizeof(float));

   // Allocate state and param memory
   cudaMalloc(&d_states, numGenerators * sizeof(curandStateMtgp32));
   cudaMalloc(&d_params, numGenerators * sizeof(mtgp32_kernel_params_t));


   // Create local param buffer of correct type
   mtgp32_kernel_params_t *h_params = new mtgp32_kernel_params_t[numGenerators];
   curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, h_params);
   cudaMemcpy(d_params, h_params, numGenerators * sizeof(mtgp32_kernel_params_t),
              cudaMemcpyHostToDevice);
   delete[] h_params;

   curandMakeMTGP32KernelState(d_states, mtgp32dc_params_fast_11213, d_params, numGenerators, seed);

   // Pre-fill both buffers
   fillBuffer(0);
   fillBuffer(1);
}

AsyncMT_d::~AsyncMT_d()
{
   cudaFree(buffers[0]);
   cudaFree(buffers[1]);
   cudaFree(d_states);
   cudaFree(d_params);
   cudaStreamDestroy(stream);
}

float *AsyncMT_d::requestSegment()
{
   if (segmentIndex >= totalSegments) {
      // Switch buffer and launch async refill on the now-unused one
      int refillBuffer = currentBuffer;
      currentBuffer = 1 - currentBuffer;
      segmentIndex = 0;
      cudaStreamSynchronize(stream);   // Ensure refillBuffer is done
      fillBuffer(refillBuffer);
   }

   float *segmentPtr = buffers[currentBuffer] + segmentIndex * segmentSize;
   segmentIndex++;
   return segmentPtr;
}

void AsyncMT_d::fillBuffer(int bufferIndex)
{
   dim3 blocks(numGenerators);
   dim3 threads(256);
   generateKernel<<<blocks, threads, 0, stream>>>(d_states, buffers[bufferIndex],
                                                  bufferSize / numGenerators);
}
