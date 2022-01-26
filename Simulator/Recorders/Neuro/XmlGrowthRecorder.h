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

#include <fstream>

#include "Model.h"
#include "XmlRecorder.h"

class XmlGrowthRecorder : public XmlRecorder {
	public:
		/// THe constructor and destructor
		XmlGrowthRecorder();

		~XmlGrowthRecorder() override;

		static IRecorder* Create() { return new XmlGrowthRecorder(); }

		/// Init radii and rates history matrices with default values
		void initDefaultValues() override;

		/// Init radii and rates history matrices with current radii and rates
		void initValues() override;

		/// Get the current radii and rates vlaues
		void getValues() override;

		/// Compile history information in every epoch
		///
		/// @param[in] neurons   The entire list of neurons.
		void compileHistories(AllVertices& neurons) override;

		/// Writes simulation results to an output destination.
		///
		/// @param  neurons the Neuron list to search from.
		void saveSimData(const AllVertices& neurons) override;

		///  Prints out all parameters to logging file.
		///  Registered to OperationManager as Operation::printParameters
		void printParameters() override;

	private:
		// TODO: There seems to be multiple copies of this in different classes...
		void getStarterNeuronMatrix(VectorMatrix& matrix, const bool* starterMap);

		// track firing rate
		CompleteMatrix ratesHistory_;

		// track radii
		CompleteMatrix radiiHistory_;

};
