/**
 * @file ISInput.h
 * 
 * @ingroup Simulator/Utils/Inputs
 *
 * @brief An interface for stimulus input classes.
 *
 * The ISInput provides an interface for stimulus input classes.
 */

#pragma once

#include "Core/Model.h"
#include "Global.h"
#include "Simulator.h"
#include "tinyxml.h"

class ISInput {
public:
   virtual ~ISInput() = default;

   /// Initialize data
   virtual void init() = 0;

   /// Terminate process
   virtual void term() = 0;

   /// Process input stimulus for each time step
   ///
   /// @param[in] psi       Pointer to the simulation information.
   virtual void inputStimulus() = 0;
};
