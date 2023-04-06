/**
 * @file SInputRegular.h
 * 
 * @ingroup Simulator/Utils/Inputs
 *
 * @brief A class that performs stimulus input (implementation Regular).
 *
 * The SInputRegular performs providing stimulus input to the network for each time step.
 * Inputs are series of current pulses, which are characterized by a duration, an interval
 * and input values.
 *
 * This class is the base class of GpuSInputRegular and HostSInputRegular.
 */

#pragma once

#include <vector>

#ifndef _SINPUTREGULAR_H_
   #define _SINPUTREGULAR_H_

   #include "ISInput.h"

class SInputRegular : public ISInput {
public:
   //! The constructor for SInputRegular.
   SInputRegular(TiXmlElement *parms);

   ~SInputRegular() = default;

   //! Initialize data.
   virtual void init();

   //! Terminate process.
   virtual void term();

protected:
   //! True if stimuls input is on.
   bool fSInput;

   //! Duration of a pulse in second.
   BGFLOAT duration;

   //! Interval between pulses in second.
   BGFLOAT interval;

   //! The number of time steps for one cycle of a stimulation
   int nStepsCycle;

   //! The time step within a cycle of stimulation
   int nStepsInCycle;

   //! The number of time steps for duration of a pulse.
   int nStepsDuration;

   //! The number of time steps for interval between pulses.
   int nStepsInterval;

   //! Initial input values
   vector<BGFLOAT> initValues;

   //! Input values, where each entry corresponds with a summationPoint.
   vector<BGFLOAT> values;

   //! Shift values, which determin the synch of stimuli (all 0 when synchronous)
   vector<int> nShiftValues;
};

#endif   // _SINPUTREGULAR_H_
