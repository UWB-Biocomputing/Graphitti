/**
 * @file GpuSInputPoisson.h
 *
 * @ingroup Simulator/Utils/Inputs
 * 
 * @brief A class that performs stimulus input (implementation Poisson on GPU).
 *
 * The GpuSInputPoisson performs providing stimulus input to the network for each time step.
 * In this version, a layer of synapses are added, which accept external spike trains.
 * Each synapse gets an indivisual spike train (Poisson distribution) characterized
 * by mean firing rate, and each synapse has individual weight value.
 */

#pragma once

#ifndef _GPUSINPUTPOISSON_H_
#define _GPUSINPUTPOISSON_H_

#include "SInputPoisson.h"
#include "GPUModel.h"
#include "AllSynapsesDeviceFuncs.h"

class GpuSInputPoisson : public SInputPoisson
{
public:
    //! The constructor for GpuSInputPoisson.
    GpuSInputPoisson(TiXmlElement* parms);
    ~GpuSInputPoisson();

    //! Initialize data.
    virtual void init();

    //! Terminate process.
    virtual void term();

    //! Process input stimulus for each time step.
    virtual void inputStimulus();

private:
    //! Allocate GPU device memory and copy values
    void allocDeviceValues(Model* model, int *deviceInteralCounter_);

    //! Dellocate GPU device memory
    void deleteDeviceValues(Model* model);

    //! Synapse structures in device memory.
    AllDSSynapsesDeviceProperties* allEdgesDevice;
 
    //! Pointer to synapse index map in device memory.
    EdgeIndexMap* edgeIndexMapDevice;

    //! Pointer to device interval counter.
    int* deviceInteralCounter_;

    //! Pointer to device masks for stimulus input
    bool* deviceMasks_;
};

#if defined(__CUDACC__)
//! Device function that processes input stimulus for each time step.
extern __global__ void inputStimulusDevice( int n, int* deviceInteralCounter_, bool* deviceMasks_, BGFLOAT deltaT, BGFLOAT lambda, curandState* devStates_d, AllDSSynapsesDeviceProperties* allEdgesDevice );
extern __global__ void applyI2SummationMap( int n, BGFLOAT* summationPoint_d, AllDSSynapsesDeviceProperties* allEdgesDevice );
extern __global__ void setupSeeds( int n, curandState* devStates_d, unsigned long seed );
#endif

#endif // _GPUSINPUTPOISSON_H_
