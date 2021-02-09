/**
 * @file GpuSInputRegular.h
 * 
 * @ingroup Simulation/Utils/Inputs
 *
 * @brief A class that performs stimulus input (implementation Regular on GPU).
 * 
 ** The GpuSInputRegular performs providing stimulus input to the network for each time step on GPU.
 ** Inputs are series of current pulses, which are characterized by a duration, an interval
 ** and input values.
 **/

#pragma once

#ifndef _GPUSINPUTREGULAR_H_
#define _GPUSINPUTREGULAR_H_

#include "SInputRegular.h"

class GpuSInputRegular : public SInputRegular
{
public:
    //! The constructor for SInputRegular.
    GpuSInputRegular(TiXmlElement* parms);
    ~GpuSInputRegular();

    //! Initialize data.
    virtual void init();

    //! Terminate process.
    virtual void term();

    //! Process input stimulus for each time step.
    virtual void inputStimulus();
};

//! Device function that processes input stimulus for each time step.
#if defined(__CUDACC__)
extern __global__ void inputStimulusDevice( int n, BGFLOAT* summationPoint_d, BGFLOAT* initValues_d, int* nShiftValues_d, int nStepsInCycle, int nStepsCycle, int nStepsDuration );
#endif

#endif // _GPUSINPUTREGULAR_H_
