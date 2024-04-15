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

#include "AllVertices.h"   //remove it after implementing the HDF5Recorder
#include "RecordableBase.h"
#include <fstream>
#include <log4cplus/loggingmacros.h>
#include <variant>
#include <vector>


/// a list of pre-defined basic data types in recorded variables
using multipleTypes = variant<uint64_t, bool, int, BGFLOAT>;

//TODO: remove it after implemtating the HDF5Recorder
class AllVertices;
class Recorder {
public:
   /// The recorded variable Type/Updated frequency
   enum UpdatedType {
      CONSTANT,   // value doesn't change in each epoch
      DYNAMIC     // value is updated in each peoch
      // Add more variable types as needed
   };
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

   /// Receives a recorded variable entity from the variable owner class
   /**
   * @brief Registers a single instance of a class derived from RecordableBase.
   * @param varName Name of the recorded variable.
   * @param recordVar Reference to the recorded variable.
   * @param variableType Updated frequency of the recorded variable.
   */
   virtual void registerVariable(const string &varName, RecordableBase &recordVar,
                                 UpdatedType variableType)
      = 0;

   /// Register a vector of instance of a class derived from RecordableBase.
   virtual void registerVariable(const string &varName, vector<RecordableBase *> &recordVars,
                                 UpdatedType variableType)
      = 0;

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