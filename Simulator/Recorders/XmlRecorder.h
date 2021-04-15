/**
 * @file XmlRecorder.h
 * 
 * @ingroup Simulator/Recorders
 *
 * @brief An implementation for recording spikes history on xml file
 * 
 * The XmlRecorder provides a mechanism for recording neuron's layout, spikes history,
 * and compile history information on xml file:
 *     -# neuron's locations, and type map,
 *     -# individual neuron's spike rate in epochs,
 *     -# network wide burstiness index data in 1s bins,
 *     -# network wide spike count in 10ms bins.
 */

#pragma once

#include <fstream>

#include "Recorders/IRecorder.h"
#include "Core/Model.h"

class XmlRecorder : public IRecorder {
public:
   /// constructor
   XmlRecorder();

   /// destructor
   ~XmlRecorder();

   /// Initialize data
   /// Create a new xml file.
   virtual void init();

   /// Init radii and rates history matrices with default values
   virtual void initDefaultValues();

   /// Init radii and rates history matrices with current radii and rates
   virtual void initValues();

   /// Get the current radii and rates vlaues
   virtual void getValues();

   /// Terminate process
   virtual void term();

   /// Compile history information in every epoch
   /// @param[in] vertices   The entire list of vertices.
   virtual void compileHistories(IAllVertices &vertices) = 0;

   /// Writes simulation results to an output destination.
   /// @param  vertices the Vertex list to search from.
   virtual void saveSimData(const IAllVertices &vertices) = 0;

   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() = 0;

protected:
   // a file stream for xml output
   ofstream stateOut_;
};

