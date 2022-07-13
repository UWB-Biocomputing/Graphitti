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
#include "EventBuffer.h"
#include "Global.h"
#include "IRecorder.h"
#include "Model.h"
#include <fstream>
#include <vector>

class XmlRecorder : public IRecorder {
public:
   // constructor which opens the xml file to store results
   XmlRecorder();

   static IRecorder *Create()
   {
      return new XmlRecorder();
   }

   /// Initialize data in the newly loadeded xml file
   virtual void init() override;

   /// Init radii and rates history matrices with default values
   virtual void initDefaultValues() override;

   /// Init radii and rates history matrices with current radii and rates
   virtual void initValues() override;

   /// Get the current radii and rates vlaues
   virtual void getValues() override;

   /// Terminate process
   virtual void term() override;

   /// Compile history information in every epoch
   /// @param[in] vertices   The entire list of vertices.
   virtual void compileHistories(AllVertices &vertices) override;

   /// Writes simulation results to an output destination.
   /// @param  vertices the Vertex list to search from.
   virtual void saveSimData(const AllVertices &vertices) override;

   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() override;

protected:
   // a file stream for xml output
   ofstream resultOut_;

   // burstiness Histogram goes through the
   VectorMatrix burstinessHist_;

   // spikes history - history of accumulated spikes count of all neurons (10 ms bin)
   VectorMatrix spikesHistory_;

   // Populates Starter neuron matrix based with boolean values based on starterMap state
   ///@param[in] matrix  starter neuron matrix
   ///@param starterMap  Bool map to reference neuron matrix location from.
   virtual void getStarterNeuronMatrix(VectorMatrix &matrix, const bool *starterMap) override;
};
