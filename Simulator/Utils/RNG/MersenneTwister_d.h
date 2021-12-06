/**
 * @file MersenneTwister_d.h
 * 
 * @ingroup Simulator/Utils/RNG
 * 
 * @brief MersenneTwister logic from Nvidia
 * 
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
// #ifndef MERSENNETWISTER_H
//#define MERSENNETWISTER_H
#ifndef mersennetwister_h
#define mersennetwister_h



#define      DCMT_SEED 4172
#define  MT_RNG_PERIOD 607


typedef struct{
    unsigned int matrix_a;
    unsigned int mask_b;
    unsigned int mask_c;
///////////////////////////////////////////////////////////////////////////////////////////////////////
//seed is nolonger used. it is replaced with iState so that the fread continues to work with the same 
//.dat file and i am able to pass the iState of each thread to the kernel without creating another global
//with more memory usage along with the garbage unused seed.
//    unsigned int seed;
	unsigned int iState;
///////////////////////////////////////////////////////////////////////////////////////////////////////
} mt_struct_stripped;


#define MT_DATAFILE "RuntimeFiles/Data/MersenneTwister.dat"
//#define   MT_RNG_COUNT 4096
#define   MT_RNG_COUNT 2500	//max threads 
#define          MT_MM 9
#define          MT_NN 19
#define       MT_WMASK 0xFFFFFFFFU
#define       MT_UMASK 0xFFFFFFFEU
#define       MT_LMASK 0x1U
#define      MT_SHIFT0 12
#define      MT_SHIFTB 7
#define      MT_SHIFTC 15
#define      MT_SHIFT1 18



#endif
//#endif
