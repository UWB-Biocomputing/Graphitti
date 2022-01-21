/**
 * @file BGTypes.h
 *
 * @brief Used to define uniform data type sizes based for all operating systems. Also used to test the speed of
 * simulation based on the data types.
 *
 * @ingroup Simulator/Utils
 *
 *
 * This type is used to measure the difference between
 * IEEE Standard 754 single and double-precision floating
 * point values.
 *
 * Single-precision (float) calculations are fast, but only
 * 23 bits are available to store the decimal.
 *
 * Double-precision (double) calculations are an order of magnitude
 * slower, but 52 bits are available to store the decimal.
 *
 * We'd like to avoid doubles, if the simulation output doesn't suffer.
 *
 *
 * For floats, uncomment the following two lines and comment DOUBLEPRECISION and
 * the other #define BGFLOAT; vice-versa for doubles.
 */

#pragma once
#include <cstdint>

#define SINGLEPRECISION
#define BGFLOAT float
//#define DOUBLEPRECISION
//#define BGFLOAT double

using PBGFloat = BGFLOAT*;

// TIMEFLOAT is used by the GPU code and needs to be a double
#define TIMEFLOAT double

// AMP
#ifdef USE_AMP
#define GPU_COMPAT_BOOL uint32_t
#else
#define GPU_COMPAT_BOOL bool
#endif // AMP

// The type for using array indexes (issue #142).
#define BGSIZE uint32_t
//#define BGSIZE uint64_t
