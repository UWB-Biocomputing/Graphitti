/**
 * @file Xml911Recorder.h
 * 
 * @ingroup Simulator/Recorders
 *
 * @brief Header file for Xml911Recorder.h
 *
 * The Xml911Recorder provides a mechanism for recording vertex's layout,
 * and compile history information on xml file:
 *     -# vertex's locations, and type map,
 */

#pragma once

#include <fstream>

#include "Recorders/XmlRecorder.h"
#include "Core/Model.h"

class Xml911Recorder : public XmlRecorder {
public:
   /// The constructor and destructor
   Xml911Recorder();

   ~Xml911Recorder();

   static IRecorder* Create() { return new Xml911Recorder(); }

   /// Init radii and rates history matrices with default values
   virtual void initDefaultValues();

   /// Init radii and rates history matrices with current radii and rates
   virtual void initValues();

   /// Get the current radii and rates vlaues
   virtual void getValues();

   /// Compile history information in every epoch
   ///
   /// @param[in] vertices   The entire list of vertices.
   virtual void compileHistories(IAllVertices &vertices);

   /// Writes simulation results to an output destination.
   ///
   /// @param  vertices the Neuron list to search from.
   virtual void saveSimData(const IAllVertices &vertices);

   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters();

private:
};
