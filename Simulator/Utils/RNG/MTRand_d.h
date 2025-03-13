/** 
* @file MTRand_d.h
*
* @ingroup Simulator/Utils/RNG
*
* @brief Mersenne Twister logic from Nvidia
*
* This file has been modified by the UW Bothell Graphitti group,
* mostly to reorganize it and make it look more like typical C++
* code. This includes splitting it into a .h and .cpp (instead of
* having everything in a .h file), and replacing enums previously
* used to define constants with consts. Given that this was designed
* to produce 32-bit random numbers, and have 32-bit internal state,
* the type uint32_t has been used throughout for precision of
* definition (now that compilers often use 64-bit ints).
*
* Mersenne Twister random number generator -- a C++ class MTRand_d
* Based on code by Makoto Matsumoto, Takuji Nishimura, and Shawn Cokus
* Richard J. Wagner  v1.0  15 May 2003  rjwagner@writeme.com
*
* The Mersenne Twister is an algorithm for generating random numbers.  It
* was designed with consideration of the flaws in various other generators.
* The period, 2^19937-1, and the order of equidistribution, 623 dimensions,
* are far greater.  The generator is also fast; it avoids multiplication and
* division, and it benefits from caches and pipelines.  For more information
* see the inventors' web page at http://www.math.keio.ac.jp/~matumoto/emt.html
*
* Reference
* M. Matsumoto and T. Nishimura, "Mersenne Twister: A 623-Dimensionally
* Equidistributed Uniform Pseudo-Random Number Generator", ACM Transactions on
* Modeling and Computer Simulation, Vol. 8, No. 1, January 1998, pp 3-30.
*
* Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
* Copyright (C) 2000 - 2003, Richard J. Wagner
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*
*   1. Redistributions of source code must retain the above copyright
*      notice, this list of conditions and the following disclaimer.
*
*   2. Redistributions in binary form must reproduce the above copyright
*      notice, this list of conditions and the following disclaimer in the
*      documentation and/or other materials provided with the distribution.
*
*   3. The names of its contributors may not be used to endorse or promote
*      products derived from this software without specific prior written
*      permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
* A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
* LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
* NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* The original code included the following notice:
*
*     When you use this, send an email to: matumoto@math.keio.ac.jp
*     with an appropriate reference to your work.
*
* It would be nice to CC: rjwagner@writeme.com and Cokus@math.washington.edu
* when you write.
*
* Not thread safe (unless auto-initialization is avoided and each thread has
* its own MTRand_d object)
*/

#pragma once

#include "BGTypes.h"   // for BGFLOAT
#include <climits>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <stdint.h>
// cereal
#include <cereal/types/polymorphic.hpp>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>

class MTRand_d {
public:
   static constexpr const char *MT_DATAFILE = "RuntimeFiles/Data/MersenneTwister_16384.dat";

   static constexpr std::size_t RNG_COUNT = 16384;   // max threads
   static constexpr std::size_t MM = 9;
   static constexpr std::size_t NN = 19;

   static constexpr std::uint32_t WMASK = 0xFFFFFFFFU;
   static constexpr std::uint32_t UMASK = 0xFFFFFFFEU;
   static constexpr std::uint32_t LMASK = 0x1U;

   static constexpr int SHIFT0 = 12;
   static constexpr int SHIFTB = 7;
   static constexpr int SHIFTC = 15;
   static constexpr int SHIFT1 = 18;

   // Constants related to DCMT and period
   static constexpr std::uint32_t DCMT_SEED = 4172;
   static constexpr std::uint32_t MT_RNG_PERIOD = 607;

   static constexpr float PI = 3.14159265358979f;

   struct mt_struct_stripped {
      unsigned int matrix_a;
      unsigned int mask_b;
      unsigned int mask_c;
      unsigned int iState;   // Replaces seed
   };

private:
   mt_struct_stripped *MT_;    // Host state
   mt_struct_stripped *MT_d;   // Device state
   unsigned int *mt_d;         // Device state values

   unsigned int mt_rng_count_;
   unsigned int mt_blocks_;
   unsigned int mt_threads_;
   unsigned int mt_nPerRng_;
   unsigned int mt_totalVertices_;
   unsigned int
      mt_noiseSize_;   //integer multiple of totalVertices for the size of the noise buffers
   unsigned int mt_seed_;
   float *mtNoise1_d;
   float *mtNoise2_d;

   void loadState();   // Function to load state from file
   void loadMTGPU(const char *fname);
   void seedMTGPU(unsigned int seed);
   void uniformMTGPU(float *d_random);
   void normalMTGPU(float *d_random);
   void initMTGPU(unsigned int seed, unsigned int blocks, unsigned int threads,
                  unsigned int nPerRng, unsigned int mt_rng_c);

   //Methods
public:
   MTRand_d();

   ~MTRand_d();

   void allocDeviceStruct();
   void deleteDeviceStruct();
   void allocHostStruct();
};
