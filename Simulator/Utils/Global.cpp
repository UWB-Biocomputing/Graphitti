/**
 *	@file Global.cpp
 *
 *	@ingroup Simulator/Utils
 *
 *  @brief Globally available functions/variables and default parameter values.
 */
#include "Global.h"
#include "MTRand.h"
#include "Norm.h"

// Debugging log data and routines
// see "global.h" for bitmask usage of debug outputs
int g_debug_mask
#if DEBUG_OUT
   = DEBUG_LOG_LOW;
#else
   = 0;
#endif

///  Converts the given index to a string with the indexes of a two-dimensional array.
///  @param  i   index to be converted.
///  @param  width   width of the two-dimensional array
///  @param  height  height of the two-dimensional array
///  @return string with the converted indexes and square brackets surrounding them.
string index2dToString(int i, int width, int height)
{
   stringstream ss;
   ss << "[" << i % width << "][" << i / height << "]";
   return ss.str();
}

///  Takes the two given coordinates and outputs them with brackets.
///  @param  x   x coordinate.
///  @param  y   y coordinate.
///  @return returns the given coordinates surrounded by square brackets.
string coordToString(int x, int y)
{
   stringstream ss;
   ss << "[" << x << "][" << y << "]";
   return ss.str();
}

///  Takes the three given coordinates and outputs them with brackets.
///  @param  x   x coordinate.
///  @param  y   y coordinate.
///  @param  z   z coordinate.
///  @return returns the given coordinates surrounded by square brackets.
string coordToString(int x, int y, int z)
{
   stringstream ss;
   ss << "[" << x << "][" << y << "][" << z << "]";
   return ss.str();
}

// MODEL INDEPENDENT FUNCTION NMV-BEGIN {
string neuronTypeToString(vertexType t)
{
   switch (t) {
      case INH:
         return "INH";
      case EXC:
         return "EXC";
      default:
         cerr << "ERROR->neuronTypeToString() failed, unknown type: " << t << endl;
         assert(false);
         return nullptr;   // Must return a value -- this will probably cascade to another failure
   }
}
// } NMV-END
#if defined(USE_GPU)
//! CUDA device ID
int g_deviceId = 0;
#endif   // USE_GPU

// A random number generator.
MTRand initRNG;

// A normalized random number generator.
unique_ptr<MTRand> noiseRNG;

//		simulation vars
uint64_t g_simulationStep = 0;

//const BGFLOAT g_synapseStrengthAdjustmentConstant = 1.0e-8;

// 		misc constants
const BGFLOAT pi = 3.1415926536;

#ifdef PERFORMANCE_METRICS
// All times in seconds
double t_host_initialization_layout;
double t_host_initialization_connections;
double t_host_advance;
double t_host_adjustEdges;

double t_gpu_rndGeneration;
double t_gpu_advanceNeurons;
double t_gpu_advanceSynapses;
double t_gpu_calcSummation;

void printPerformanceMetrics(const float total_time, int steps)
{
   cout << "t_gpu_rndGeneration: " << t_gpu_rndGeneration << " ms ("
        << t_gpu_rndGeneration / total_time * 100 << "%)" << endl;
   cout << "t_gpu_advanceNeurons: " << t_gpu_advanceNeurons << " ms ("
        << t_gpu_advanceNeurons / total_time * 100 << "%)" << endl;
   cout << "t_gpu_advanceSynapses: " << t_gpu_advanceSynapses << " ms ("
        << t_gpu_advanceSynapses / total_time * 100 << "%)" << endl;
   cout << "t_gpu_calcSummation: " << t_gpu_calcSummation << " ms ("
        << t_gpu_calcSummation / total_time * 100 << "%)" << endl;

   cout << "\nHost initialization (layout): " << t_host_initialization_layout << " seconds ("
        << t_host_initialization_layout / total_time * 100 << "%)" << endl;

   cout << "\nHost initialization (connections): " << t_host_initialization_connections
        << " seconds (" << t_host_initialization_connections / total_time * 100 << "%)" << endl;

   cout << "\nHost advance: " << t_host_advance << " seconds (" << t_host_advance / total_time * 100
        << "%)" << endl;

   cout << "\nHost adjustEdges: " << t_host_adjustEdges << " seconds ("
        << t_host_adjustEdges / total_time * 100 << "%)" << endl;

   cout << "\nAverage time per simulation epoch:" << endl;

   cout << "t_gpu_rndGeneration: " << t_gpu_rndGeneration / steps << " ms/epoch" << endl;
   cout << "t_gpu_advanceNeurons: " << t_gpu_advanceNeurons / steps << " ms/epoch" << endl;
   cout << "t_gpu_advanceSynapses: " << t_gpu_advanceSynapses / steps << " ms/epoch" << endl;
   cout << "t_gpu_calcSummation: " << t_gpu_calcSummation / steps << " ms/epoch" << endl;
}
#endif   // PERFORMANCE_METRICS

// TODO comment
const string MATRIX_TYPE = "complete";
// TODO comment
const string MATRIX_INIT = "const";
