/**
 * @file XmlGrowthRecorder.h
 * 
 * @ingroup Simulation/Recorders
 *
 * @brief Header file for XmlGrowthRecorder.h
 *
 * The XmlGrowthRecorder provides a mechanism for recording neuron's layout, spikes history,
 * and compile history information on xml file:
 *     -# neuron's locations, and type map,
 *     -# individual neuron's spike rate in epochs,
 *     -# network wide burstiness index data in 1s bins,
 *     -# network wide spike count in 10ms bins,
 *     -# individual neuron's radius history of every epoch.
 */

#pragma once

#include <fstream>

#include "Recorders/XmlRecorder.h"
#include "Core/Model.h"

class XmlGrowthRecorder : public XmlRecorder {
public:
   /// THe constructor and destructor
   XmlGrowthRecorder();

   ~XmlGrowthRecorder();

   static IRecorder* Create() { return new XmlGrowthRecorder(); }

   /// Init radii and rates history matrices with default values
   virtual void initDefaultValues();

   /// Init radii and rates history matrices with current radii and rates
   virtual void initValues();

   /// Get the current radii and rates vlaues
   virtual void getValues();

   /// Compile history information in every epoch
   ///
   /// @param[in] neurons   The entire list of neurons.
   virtual void compileHistories(IAllVertices &neurons);

   /// Writes simulation results to an output destination.
   ///
   /// @param  neurons the Neuron list to search from.
   virtual void saveSimData(const IAllVertices &neurons);

   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters();

private:
   // track firing rate
   CompleteMatrix ratesHistory_;

   // track radii
   CompleteMatrix radiiHistory_;
};

