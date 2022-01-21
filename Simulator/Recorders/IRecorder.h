/**
 * @file IRecorder.h
 *
 * @ingroup Simulator/Recorders
 *
 * @brief An interface for recording spikes history
 */

#pragma once

#include <string>

#include <log4cplus/loggingmacros.h>

#include "Global.h"
#include "Model.h"
#include "Simulator.h"
#include "AllVertices.h"

class Model;
class AllVertices;

class IRecorder {
public:
   virtual ~IRecorder() {}

   /// Initialize data
   ///
   /// @param[in] stateOutputFileName       File name to save histories
   virtual void init() = 0;

   /// Init radii and rates history matrices with default values
   virtual void initDefaultValues() = 0;

   /// Init radii and rates history matrices with current radii and rates
   virtual void initValues() = 0;

   /// Get the current radii and rates values
   virtual void getValues() = 0;

   /// Terminate process
   virtual void term() = 0;

   /// Compile history information in every epoch
   ///
   /// @param[in] neurons   The entire list of neurons.
   virtual void compileHistories(AllVertices &vertices) = 0;

   /// Writes simulation results to an output destination.
   ///
   /// @param[in] neurons   The entire list of neurons.
   virtual void saveSimData(const AllVertices &vertices) = 0;

   /// Prints loaded parameters to logging file.
   virtual void printParameters() = 0;

protected:
   /// File path to the file that the results will be printed to.
   std::string resultFileName_;

   /// Loggers used to print to using log4cplus logging macros, prints to Results/Debug/logging.txt
   log4cplus::Logger fileLogger_;
};
