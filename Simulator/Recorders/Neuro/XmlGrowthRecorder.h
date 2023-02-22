/**
 * @file XmlGrowthRecorder.h
 * 
 * @ingroup Simulator/Recorders
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
 *
 */

#pragma once

#include "Model.h"
#include "XmlRecorder.h"
#include <fstream>

class XmlGrowthRecorder : public XmlRecorder {
public:
   /// THe constructor and destructor
   XmlGrowthRecorder();

   ~XmlGrowthRecorder();

   static IRecorder *Create()
   {
      return new XmlGrowthRecorder();
   }

   /// Init radii and rates history matrices with default values
   virtual void initDefaultValues() override;

   /// Initialize data in the newly loadeded xml file
   virtual void init() override;

   /// Init radii and rates history matrices with current radii and rates
   virtual void initValues() override;

   /// Get the current radii and rates vlaues
   virtual void getValues() override;

   /// Compile history information in every epoch
   ///
   /// @param[in] neurons   The entire list of neurons.
   virtual void compileHistories(AllVertices &neurons) override;

   /// Writes simulation results to an output destination.
   ///
   /// @param  neurons the Neuron list to search from.
   virtual void saveSimData(const AllVertices &neurons) override;

   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() override;

private:
   // TODO: There seems to be multiple copies of this in different classes...
   void getStarterNeuronMatrix(VectorMatrix &matrix, const std::vector<bool> &starterMap);

   // track firing rate
   shared_ptr<CompleteMatrix> ratesHistory_;

   // track radii
   shared_ptr<CompleteMatrix> radiiHistory_;
};
