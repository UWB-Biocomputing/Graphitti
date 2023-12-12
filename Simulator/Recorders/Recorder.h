/**
 * @file Recorder.h
 *
 * @ingroup Simulator/Recorders
 *
 * @brief An interface for recording variables/data acquisition system
 *
 */

#pragma once
using namespace std;

#include "AllVertices.h"
#include "RecordableBase.h"
#include <fstream>
#include <log4cplus/loggingmacros.h>
#include <vector>

class AllVertices;
class Recorder {
public:
   virtual ~Recorder() = default;

   /// Initialize data
   /// @param[in] stateOutputFileName  File name to save histories
   virtual void init() = 0;

   // ToDo : remove it ?
   /// Init radii and rates history matrices with default values
   virtual void initDefaultValues() = 0;


   /// Init radii and rates history matrices with current radii and rates
   virtual void initValues() = 0;

   // ToDo : remove it ?
   /// Get the current radii and rates values
   virtual void getValues() = 0;

   /// Terminate process
   virtual void term() = 0;

   // TODO: No parameters needed (AllVertices &vertices)
   /// Compile/capture variable history information in every epoch
   virtual void compileHistories(AllVertices &vertices) = 0;

   // TODO: No parameters needed (AllVertices &vertices)
   /// Writes simulation results to an output destination.
   virtual void saveSimData(const AllVertices &vertices) = 0;

   /// Prints loaded parameters to logging file.
   virtual void printParameters() = 0;

   /// Register a single instance of a class derived from RecordableBase.
   /// It stores the address of the registered variable and the related information
   /// of this recorded variable
   virtual void registerVariable(const string &varName, RecordableBase *recordVar) = 0;

   /// register a vector of instance of a class derived from RecordableBase.
   virtual void registerVariable(const string &varName, vector<RecordableBase *> &recordVars) = 0;

protected:
   /// File path to the file that the results will be printed to.
   string resultFileName_;

   /// Loggers used to print to using log4cplus logging macros, prints to Results/Debug/logging.txt
   log4cplus::Logger fileLogger_;

   // ToDo : remove it ?
   /// Populates Starter neuron matrix based with boolean values based on starterMap state
   ///@param[in] matrix  starter neuron matrix
   ///@param starterMap  Bool map to reference neuron matrix location from.
   virtual void getStarterNeuronMatrix(VectorMatrix &matrix, const vector<bool> &starterMap) = 0;
};