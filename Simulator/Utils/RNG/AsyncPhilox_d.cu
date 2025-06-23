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

/// @brief Kernel to generate Gaussian (normal) random numbers using Philox.
///
/// Each thread loads its own Philox RNG state, generates one or more
/// random floats using a strided loop, and writes them into the output buffer.
/// The updated RNG state is written back to global device memory.
///
/// @param states      Array of Philox RNG states, one per thread.
/// @param output      Output buffer for generated random floats.
/// @param bufferSize  Total number of floats to generate (length of output buffer).
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

/// @brief Kernel to initialize Philox RNG states for each thread.
///
/// Each thread initializes its entry in the RNG state array using a fixed seed.
/// This is typically called once before generating random numbers.
///
/// @param states        Array to hold initialized Philox RNG states.
/// @param seed          Seed value used to initialize curand.
/// @param totalThreads  Total number of threads that will use RNG states.
__global__ void initPhilox(curandStatePhilox4_32_10_t *states, unsigned long seed, int totalThreads)
{
   int gid = blockIdx.x * blockDim.x + threadIdx.x;
   if (gid >= totalThreads)
      return;
   curand_init(seed, gid, 0, &states[gid]);
}

/// Initializes generator and allocates device memory
/// @param samplesPerSegment Number of total vertices
/// @param seed RNG seed.
void AsyncPhilox_d::loadAsyncPhilox(int samplesPerSegment, unsigned long seed)
{
   // hostBuffer = nullptr;
   // cudaHostAlloc(&hostBuffer, samplesPerSegment * sizeof(float), cudaHostAllocDefault);
   // logfile = std::fopen("philox_output_32_10.bin", "wb");
   //consoleLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("console"));
   segmentSize_ = samplesPerSegment;
   seed_ = seed;
   currentBuffer_ = 0;
   segmentIndex_ = 0;

   totalSegments_ = 10;

#ifdef ENABLE_NVTX
   nvtxMarker = 10000 / totalSegments;   // make a marker every nvtxMarker buffer fills;
   nvtxCurrentMarker = nvtxMarker;       // count down to color flip
#endif
   bufferSize_ = segmentSize_ * totalSegments_;
   numBlocks_ = 64;   //placeholder num of blocks
   numThreads_ = 64;

   totalThreads_ = numThreads_ * numBlocks_;

   int leastPriority, greatestPriority;
   HANDLE_ERROR(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
   // └─ leastPriority is the numerically largest value → lowest actual priority
   // └─ greatestPriority is the numerically smallest value → highest actual priority

   // Create internal stream
   HANDLE_ERROR(cudaStreamCreateWithPriority(&RNG_stream_, cudaStreamNonBlocking, leastPriority));

   // Allocate two large buffers
   HANDLE_ERROR(cudaMalloc(&buffers_d[0], bufferSize_ * sizeof(float)));
   HANDLE_ERROR(cudaMalloc(&buffers_d[1], bufferSize_ * sizeof(float)));

   HANDLE_ERROR(cudaMalloc(&spStates_d, totalThreads_ * sizeof(curandStatePhilox4_32_10_t)));

   initPhilox<<<totalThreads_ + 255 / 256, 256, 0, RNG_stream_>>>(spStates_d, seed_, totalThreads_);

   // Pre-fill both buffers
   fillBuffer(0);
   fillBuffer(1);
   HANDLE_ERROR(cudaStreamSynchronize(
      RNG_stream_));   //wait for both buffers to be filled before the first request
}

/// Free device memory
void AsyncPhilox_d::deleteDeviceStruct()
{
   // std::fclose(logfile);
   // cudaFree(hostBuffer);
   HANDLE_ERROR(cudaFree(buffers_d[0]));
   HANDLE_ERROR(cudaFree(buffers_d[1]));
   HANDLE_ERROR(cudaFree(spStates_d));

   HANDLE_ERROR(cudaStreamDestroy(RNG_stream_));
}

AsyncPhilox_d::~AsyncPhilox_d()
{
}

/// Request a new segment of generated noise.
/// @return Pointer to a slice of device memory containing noise.
float *AsyncPhilox_d::requestSegment()
{
   //LOG4CPLUS_TRACE(consoleLogger_, "request segment");
   //auto start = std::chrono::high_resolution_clock::now();
#ifdef ENABLE_NVTX
   static bool flipColor;
#endif
   if (segmentIndex_ >= totalSegments_) {
      // Switch buffer and launch async refill on the now-unused one

#ifdef ENABLE_NVTX
      if (nvtxCurrentMarker_ <= 0) {
         nvtxPop();
         if (flipColor == true)
            nvtxPushColor("10,000 time steps", Color::RED);
         else
            nvtxPushColor("10,000 time steps", Color::BLUE);

         flipColor = !flipColor;
         nvtxCurrentMarker_ = nvtxMarker_;
      } else
         --nvtxCurrentMarker_;
#endif

      int refillBuffer = currentBuffer_;
      currentBuffer_ = 1 - currentBuffer_;
      segmentIndex_ = 0;
      cudaStreamSynchronize(RNG_stream_);   // Ensure refillBuffer is done
      fillBuffer(refillBuffer);
   }

   float *segmentPtr = buffers_d[currentBuffer_] + segmentIndex_ * segmentSize_;
   segmentIndex_ += 1;

   // auto end = std::chrono::high_resolution_clock::now();
   // std::cout << "Segment: " << segmentIndex << ", Launch time: " << (end - start).count() << " ns\n";
   // cudaMemcpy(hostBuffer, segmentPtr, segmentSize * sizeof(float), cudaMemcpyDeviceToHost);
   // std::fwrite(hostBuffer, sizeof(float), segmentSize, logfile);

   return segmentPtr;
}

/// Internal helper to fill a specified buffer with random floats.
/// @param bufferIndex Index (0 or 1) of the buffer to fill.
void AsyncPhilox_d::fillBuffer(int bufferIndex)
{
   //LOG4CPLUS_TRACE(consoleLogger_, "filling buffer:");
   generatePhilox<<<numBlocks_, numThreads_, 0, RNG_stream_>>>(spStates_d, buffers_d[bufferIndex],
                                                               bufferSize_);
}
