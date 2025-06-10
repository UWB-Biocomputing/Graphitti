/**
 * @file AsyncPhilox_d.cu
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

#include "AsyncPhilox_d.h"
#include "NvtxHelper.h"
#include <cassert>
#include <chrono>
#include <iostream>

__global__ void generatePhilox(curandStatePhilox4_32_10_t *states, float *output, int bufferSize)
{
   // Compute a unique global index for this thread
   int threadId = threadIdx.x;
   int blockId = blockIdx.x;
   int threadsPerBlock = blockDim.x;
   int totalThreads = gridDim.x * threadsPerBlock;
   int gid = blockId * threadsPerBlock + threadId;

   // Load this thread’s Philox state
   curandStatePhilox4_32_10_t local = states[gid];

   // Stride‐loop: write one random per iteration until we cover bufferSize
   for (int idx = gid; idx < bufferSize; idx += totalThreads) {
      output[idx] = curand_normal(&local);
   }

   // Store back the updated state
   states[gid] = local;
}

__global__ void initPhilox(curandStatePhilox4_32_10_t *states, unsigned long seed, int totalThreads)
{
   int gid = blockIdx.x * blockDim.x + threadIdx.x;
   if (gid >= totalThreads)
      return;
   curand_init(seed, gid, 0, &states[gid]);
}

void AsyncPhilox_d::loadAsyncPhilox(int samplesPerSegment, unsigned long seed)
{
   // hostBuffer = nullptr;
   // cudaHostAlloc(&hostBuffer, samplesPerSegment * sizeof(float), cudaHostAllocDefault);
   // logfile = std::fopen("philox_output_32_10.bin", "wb");
   //consoleLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("console"));
   segmentSize = samplesPerSegment;
   seed = seed;
   currentBuffer = 0;
   segmentIndex = 0;

   totalSegments = 10;

#ifdef ENABLE_NVTX
   nvtxMarker = 10000 / totalSegments;   // make a marker every nvtxMarker buffer fills;
   nvtxCurrentMarker = nvtxMarker;       // count down to color flip
#endif
   bufferSize = segmentSize * totalSegments;
   numBlocks = 64;   //placeholder num of blocks
   numThreads = 64;

   totalThreads = numThreads * numBlocks;

   int leastPriority, greatestPriority;
   HANDLE_ERROR(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
   // └─ leastPriority is the numerically largest value → lowest actual priority
   // └─ greatestPriority is the numerically smallest value → highest actual priority

   // Create internal stream
   HANDLE_ERROR(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, leastPriority));

   // Allocate two large buffers
   HANDLE_ERROR(cudaMalloc(&buffers[0], bufferSize * sizeof(float)));
   HANDLE_ERROR(cudaMalloc(&buffers[1], bufferSize * sizeof(float)));

   HANDLE_ERROR(cudaMalloc(&spStates, totalThreads * sizeof(curandStatePhilox4_32_10_t)));

   initPhilox<<<totalThreads + 255 / 256, 256, 0, stream>>>(spStates, seed, totalThreads);

   // Pre-fill both buffers
   fillBuffer(0);
   fillBuffer(1);
   HANDLE_ERROR(cudaStreamSynchronize(
      stream));   //wait for both buffers to be filled before the first request
}
void AsyncPhilox_d::deleteDeviceStruct()
{
   // std::fclose(logfile);
   // cudaFree(hostBuffer);
   HANDLE_ERROR(cudaFree(buffers[0]));
   HANDLE_ERROR(cudaFree(buffers[1]));
   HANDLE_ERROR(cudaFree(spStates));

   HANDLE_ERROR(cudaStreamDestroy(stream));
}
AsyncPhilox_d::~AsyncPhilox_d()
{
}

float *AsyncPhilox_d::requestSegment()
{
   //LOG4CPLUS_TRACE(consoleLogger_, "request segment");
   //auto start = std::chrono::high_resolution_clock::now();
   static bool flipColor;
   if (segmentIndex >= totalSegments) {
      // Switch buffer and launch async refill on the now-unused one

#ifdef ENABLE_NVTX
      if (nvtxCurrentMarker <= 0) {
         nvtxPop();
         if (flipColor == true)
            nvtxPushColor("10,000 time steps", Color::RED);
         else
            nvtxPushColor("10,000 time steps", Color::BLUE);

         flipColor = !flipColor;
         nvtxCurrentMarker = nvtxMarker;
      } else
         --nvtxCurrentMarker;
#endif

      int refillBuffer = currentBuffer;
      currentBuffer = 1 - currentBuffer;
      segmentIndex = 0;
      cudaStreamSynchronize(stream);   // Ensure refillBuffer is done
      fillBuffer(refillBuffer);
   }

   float *segmentPtr = buffers[currentBuffer] + segmentIndex * segmentSize;
   segmentIndex += 1;

   // auto end = std::chrono::high_resolution_clock::now();
   // std::cout << "Segment: " << segmentIndex << ", Launch time: " << (end - start).count() << " ns\n";
   // cudaMemcpy(hostBuffer, segmentPtr, segmentSize * sizeof(float), cudaMemcpyDeviceToHost);
   // std::fwrite(hostBuffer, sizeof(float), segmentSize, logfile);

   return segmentPtr;
}

void AsyncPhilox_d::fillBuffer(int bufferIndex)
{
   //LOG4CPLUS_TRACE(consoleLogger_, "filling buffer:");
   generatePhilox<<<numBlocks, numThreads, 0, stream>>>(spStates, buffers[bufferIndex], bufferSize);
}
