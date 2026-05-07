/**
 *	@file Global.h
 *
 *	@ingroup Simulator/Utils
 *
 *	@brief Globally available functions/variables and default parameter values
 */
// Globally available functions and default parameter values.

#pragma once
#include "MTRand.h"

// Debug output is included in both debug/release builds now.
// The Default for debug is "LOW" and "OFF" for Release.

// Mask bit values:
// 0 (1) -- Normal low-level debugging
// 1 (2) -- Medium level debug info
// 2 (4) -- high/detailed level debug info
// 3 (8) -- parser XML logging
// 4 (16) -- Matrix (CompleteMatrix) debugging
// 5 (32)  -- SparseMatrix debugging
// 6 (64) --  VectorMatrix debugging
#define DEBUG_LOG_LOW 1
#define DEBUG_LOG_MID 2
#define DEBUG_LOG_HI 4
#define DEBUG_LOG_PARSER 8
#define DEBUG_LOG_MATRIX 16
#define DEBUG_LOG_SPARSE 32
#define DEBUG_LOG_VECTOR 64
#define DEBUG_LOG_SYNAPSE 128
#define DEBUG(__x) DEBUG_LOW(__x)
#define DEBUG_LOW(__x) DEBUG_LOG(DEBUG_LOG_LOW, __x)
#define DEBUG_MID(__x) DEBUG_LOG(DEBUG_LOG_MID, __x)
#define DEBUG_HI(__x) DEBUG_LOG(DEBUG_LOG_HI, __x)
#define DEBUG_PARSER(__x) DEBUG_LOG(DEBUG_LOG_PARSER, __x)
#define DEBUG_MATRIX(__x) DEBUG_LOG(DEBUG_LOG_MATRIX, __x)
#define DEBUG_SPARSE(__x) DEBUG_LOG(DEBUG_LOG_SPARSE, __x)
#define DEBUG_VECTOR(__x) DEBUG_LOG(DEBUG_LOG_VECTOR, __x)
#define DEBUG_SYNAPSE(__x) DEBUG_LOG(DEBUG_LOG_SYNAPSE, __x)
#ifdef __CUDACC__
// extern __constant__ int d_debug_mask[];
// #define DEBUG_LOG(__lvl, __x) { if(__lvl & d_debug_mask[0]) { __x } }
#else
   #define DEBUG_LOG(__lvl, __x)                                                                   \
      {                                                                                            \
         if (__lvl & g_debug_mask) {                                                               \
            __x                                                                                    \
         }                                                                                         \
      }
#endif

extern int g_debug_mask;

#include <cassert>
#include <cstdint>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>
#include "BGTypes.h"
#include "GraphProperties.h"
#include "Matrix/MatrixDefaults.h"
   //#include "Norm.h"
#include "Coordinate.h"
#include "VectorMatrix.h"
#include "VertexType.h"

// If defined, a table with time and each neuron voltage will output to stdout.
//#define DUMP_VOLTAGES

#if defined(USE_GPU)
//! CUDA device ID
extern int g_deviceId;
#endif   // USE_GPU

// The constant PI.
extern const BGFLOAT pi;

// A random number generator.
extern MTRand initRNG;

// A normalized random number generator.
extern std::unique_ptr<MTRand> noiseRNG;

// The current simulation step.
extern std::uint64_t g_simulationStep;

inline constexpr int g_nMaxChunkSize = 100;

// Edge types.
// NEURO:
//	II - Synapse from inhibitory neuron to inhibitory neuron.
//	IE - Synapse from inhibitory neuron to excitory neuron.
//	EI - Synapse from excitory neuron to inhibitory neuron.
//	EE - Synapse from excitory neuron to excitory neuron.
// NG911:
//  CP - Caller to PSAP
//  PR - PSAP to Responder
//  RC - Responder to Caller
//  PP - PSAP to PSAP

enum class edgeType {
   // NEURO
   II = 0,
   IE = 1,
   EI = 2,
   EE = 3,
   // NG911
   CP = 4,
   PR = 5,
   PC = 6,
   PP = 7,
   RP = 8,
   RC = 9,
   // UNDEF
   ETYPE_UNDEF = -1
};
// Custom streaming operator<< for the enum class edgeType
inline std::ostream &operator<<(std::ostream &os, edgeType eT)
{
   os << static_cast<int>(eT);
   return os;
}

// The default time step size.
inline constexpr double DEFAULT_dt = 1e-4;   // MODEL INDEPENDENT
// } NMV-END

// Converts a 1-d index into a coordinate string.
std::string index2dToString(int i, int width, int height);
// Converts a 2-d coordinate into a string.
std::string coordToString(int x, int y);
// Converts a 3-d coordinate into a string.
std::string coordToString(int x, int y, int z);

template <typename T> std::ostream &operator<<(std::ostream &os, const std::vector<T> &v)
{
   for (const auto &element : v) {
      os << element << " ";
   }
   return os;
}

template <typename T> std::string vectorToXML(const std::vector<T> &v, const std::string &name)
{
   std::stringstream ss;
   ss << "   <Matrix name=\"" << name << "\">\n";
   ss << "   " << v << "\n";
   ss << "   </Matrix>";
   return ss.str();
}

template <typename T>
std::string vector2dToXML(const std::vector<T> &v, const std::string &name, const std::string &rowName)
{
   std::stringstream ss;
   ss << "   <Matrix name=\"" << name << "\">\n";
   for (size_t i = 0; i < v.size(); ++i) {
      if (v[i].empty()) {
         continue;
      }   // No log to print

      ss << "      <" << rowName << " id=\"" << i << "\">\n";
      ss << "      " << v[i] << "\n";
      ss << "      </" << rowName << ">\n";
   }
   ss << "   </Matrix>";
   return ss.str();
}

#ifdef PERFORMANCE_METRICS
// All times in seconds
extern double t_host_initialization_layout;
extern double t_host_initialization_connections;
extern double t_host_advance;
extern double t_host_adjustEdges;

extern double t_gpu_rndGeneration;
extern double t_gpu_advanceNeurons;
extern double t_gpu_advanceSynapses;
extern double t_gpu_calcSummation;

void printPerformanceMetrics(const float total_time, int steps);
#endif   // PERFORMANCE_METRICS

