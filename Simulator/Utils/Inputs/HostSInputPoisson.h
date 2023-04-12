/**
 * @file HostSInputPoisson.h
 * 
 * @ingroup Simulator/Utils/Inputs
 *
 * @brief A class that performs stimulus input (implementation Poisson).
 *
 * The HostSInputPoisson performs providing stimulus input to the network for each time step.
 * In this version, a layer of synapses are added, which accept external spike trains.
 * Each synapse gets an indivisual spike train (Poisson distribution) characterized
 * by mean firing rate, and each synapse has individual weight value.
 */

#pragma once

#define _HOSTSINPUTPOISSON_H_

#include "SInputPoisson.h"

class HostSInputPoisson : public SInputPoisson {
public:
   // The constructor for HostSInputPoisson.
   HostSInputPoisson(TiXmlElement *parms);

   ~HostSInputPoisson() = default;

   // Initialize data.
   virtual void init();

   // Terminate process.
   virtual void term();

   // Process input stimulus for each time step.
   virtual void inputStimulus();

private:
};
