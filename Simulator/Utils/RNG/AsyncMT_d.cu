#include "AsyncMT_d.h"
#include <cassert>
#include <curand_mtgp32dc_p_11213.h>
#include <iostream>
#include <chrono>
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

void AsyncMT_d::loadAsyncMT(int samplesPerSegment, unsigned long seed)
{
   // hostBuffer = nullptr;
   // cudaHostAlloc(&hostBuffer, samplesPerSegment * sizeof(float), cudaHostAllocDefault);
   // logfile = std::fopen("mt_output.bin", "wb");
   //consoleLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("console"));
   segmentSize = samplesPerSegment;
   seed = seed;
   currentBuffer = 0;
   segmentIndex = 0;
   totalSegments = 10000;   // Each buffer has 10000 segments
   bufferSize = segmentSize * totalSegments;
   totalSamples = bufferSize * 2;
   numGenerators = 50;   //placeholder num of blocks

   // Create internal stream
   HANDLE_ERROR(cudaStreamCreate(&stream));

   // Allocate two large buffers
   HANDLE_ERROR(cudaMalloc(&buffers[0], bufferSize * sizeof(float)));
   HANDLE_ERROR(cudaMalloc(&buffers[1], bufferSize * sizeof(float)));

   // Allocate state and param memory
   HANDLE_ERROR(cudaMalloc(&d_states, numGenerators * sizeof(curandStateMtgp32)));
   HANDLE_ERROR(cudaMalloc(&d_params, numGenerators * sizeof(mtgp32_kernel_params_t)));


   // Create local param buffer of correct type
   mtgp32_kernel_params_t *h_params = new mtgp32_kernel_params_t[numGenerators];
   curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, h_params);
   HANDLE_ERROR(cudaMemcpy(d_params, h_params, numGenerators * sizeof(mtgp32_kernel_params_t),
              cudaMemcpyHostToDevice));
   delete[] h_params;

   curandMakeMTGP32KernelState(d_states, mtgp32dc_params_fast_11213, d_params, numGenerators, seed);

   // Pre-fill both buffers
   fillBuffer(0);
   fillBuffer(1);
   HANDLE_ERROR(cudaStreamSynchronize(stream)); //wait for both buffers to be filled before the first request
}
void AsyncMT_d::deleteDeviceStruct(){
   // std::fclose(logfile);
   // cudaFree(hostBuffer);
   HANDLE_ERROR(cudaFree(buffers[0]));
   HANDLE_ERROR(cudaFree(buffers[1]));
   HANDLE_ERROR(cudaFree(d_states));
   HANDLE_ERROR(cudaFree(d_params));
   HANDLE_ERROR(cudaStreamDestroy(stream));
}
AsyncMT_d::~AsyncMT_d()
{
}

float *AsyncMT_d::requestSegment()
{
   //LOG4CPLUS_TRACE(consoleLogger_, "request segment");
   //auto start = std::chrono::high_resolution_clock::now();
   if (segmentIndex >= totalSegments) {
      // Switch buffer and launch async refill on the now-unused one
      int refillBuffer = currentBuffer;
      currentBuffer = 1 - currentBuffer;
      segmentIndex = 0;
      cudaStreamSynchronize(stream);   // Ensure refillBuffer is done
      fillBuffer(refillBuffer);
      //cudaStreamSynchronize(stream);
   }

   float *segmentPtr = buffers[currentBuffer] + segmentIndex * segmentSize;
   segmentIndex += 1;

   // auto end = std::chrono::high_resolution_clock::now();
   // std::cout << "Segment: " << segmentIndex << ", Launch time: " << (end - start).count() << " ns\n";
   // cudaMemcpy(hostBuffer, segmentPtr, segmentSize * sizeof(float), cudaMemcpyDeviceToHost);
   // std::fwrite(hostBuffer, sizeof(float), segmentSize, logfile);
   return segmentPtr;
}

void AsyncMT_d::fillBuffer(int bufferIndex)
{
   dim3 blocks(numGenerators);
   dim3 threads(256);
   //LOG4CPLUS_TRACE(consoleLogger_, "filling buffer:");
   generateKernel<<<blocks, threads, 0, stream>>>(d_states, buffers[bufferIndex],
                                                  bufferSize / numGenerators);
}
