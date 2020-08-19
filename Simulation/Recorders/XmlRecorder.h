/*
* @file XmlRecorder.h
*
* @ingroup Core
*
* @brief An implementation for recording spikes history on xml file
*
* The XmlRecorder provides a mechanism for recording neuron's layout, spikes history,
* and compile history information on xml file:
*    -# neuron's locations, and type map,
*    -# individual neuron's spike rate in epochs,
*    -# network wide burstiness index data in 1s bins,
*    -# network wide spike count in 10ms bins.
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

   /// return pointer to new instance of this class
   static IRecorder* Create() { return new XmlRecorder(); }

   /// Initialize data
   /// Create a new xml file.
   virtual void init();

   /// Init radii and rates history matrices with default values
   virtual void initDefaultValues();

   /// Init radii and rates history matrices with current radii and rates
   virtual void initValues();

   /// Get the current radii and rates values
   virtual void getValues();

   /// Terminate process
   virtual void term();

   /// Compile history information in every epoch
   /// @param[in] neurons 	The entire list of neurons.
   virtual void compileHistories(IAllNeurons &neurons);

   /// Writes simulation results to an output destination.
   /// @param  neurons the Neuron list to search from.
   virtual void saveSimData(const IAllNeurons &neurons);

protected:
   
   /// Get starter Neuron matrix.
   /// @param  matrix: Starter Neuron matrix.
   /// @param  starter_map: reference neuron matrix location
   void getStarterNeuronMatrix(VectorMatrix &matrix, const bool *starter_map);


   ofstream stateOut;                ///< a file stream for xml output
   VectorMatrix burstinessHist;      ///< burstiness Histogram
   VectorMatrix spikesHistory;       ///< history of accumulated spikes count of all neurons (10 ms bin)
};

