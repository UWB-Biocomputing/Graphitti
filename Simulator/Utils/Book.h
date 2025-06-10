/**
 * @file Book.h
 * 
 * @ingroup Simulator/Utils
 * 
 * @brief Handles CUDA exceptions
 *
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */


#pragma once

#ifdef USE_GPU
#include <cuda_runtime.h>
#endif
#include <stdio.h>
//! CUDA Exception handler
static void HandleError(cudaError_t err, const char *file, int line)
{
   if (err != cudaSuccess) {
      printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
      exit(EXIT_FAILURE);
   }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
