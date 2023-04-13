/**
 * @file HostSInputRegular.h
 * 
 * @ingroup Simulator/Utils/Inputs
 *
 * @brief A class that performs stimulus input (implementation Regular).
 * 
 * The HostSInputRegular performs providing stimulus input to the network for each time step.
 * Inputs are series of current pulses, which are characterized by a duration, an interval
 * and input values.
 */

#pragma once

#include "SInputRegular.h"

class HostSInputRegular : public SInputRegular {
public:
   //! The constructor for HostSInputRegular.
   HostSInputRegular(TiXmlElement *parms);

   ~HostSInputRegular() = default;

   //! Initialize data.
   virtual void init();

   //! Terminate process.
   virtual void term();

   //! Process input stimulus for each time step.
   virtual void inputStimulus();

private:
};
