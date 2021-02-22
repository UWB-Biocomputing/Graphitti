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

#ifndef _HOSTSINPUTREGULAR_H_
#define _HOSTSINPUTREGULAR_H_

#include "SInputRegular.h"

class HostSInputRegular : public SInputRegular
{
public:
    //! The constructor for HostSInputRegular.
    HostSInputRegular(TiXmlElement* parms);
    ~HostSInputRegular();

    //! Initialize data.
    virtual void init();

    //! Terminate process.
    virtual void term();

    //! Process input stimulus for each time step.
    virtual void inputStimulus();

private:
};

#endif // _HOSTSINPUTREGULAR_H_
