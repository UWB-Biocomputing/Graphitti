/**
 * @file SInputPoisson.h
 *
 * @ingroup Simulator/Utils/Inputs
 *
 * @brief A class that performs stimulus input (implementation Poisson).
 *
 * The SInputPoisson performs providing stimulus input to the network for each time step.
 * In this version, a layer of synapses are added, which accept external spike trains.
 * Each synapse gets an individual spike train (Poisson distribution) characterized
 * by mean firing rate, and each synapse has an individual weight value.
 *
 * This class is the base class of GpuSInputPoisson and HostSInputPoisson.
 */

#pragma once

#ifndef _SINPUTPOISSON_H_
   #define _SINPUTPOISSON_H_

   #include "AllDSSynapses.h"
   #include "ISInput.h"

class SInputPoisson : public ISInput {
public:
   //! The constructor for SInputPoisson.
   SInputPoisson(TiXmlElement *parms);
   ~SInputPoisson();

   //! Initialize data.
   virtual void init();

   //! Terminate process.
   virtual void term();

protected:
   //! True if stimuls input is on.
   bool fSInput;

   //! synapse weight
   BGFLOAT weight;

   //! inverse firing rate
   BGFLOAT lambda;

   //! interval counter
   int *nISIs;

   //! List of synapses
   AllEdges *edges_;

   //! Masks for stimulus input
   bool *masks;
};

#endif   // _SINPUTPOISSON_H_
