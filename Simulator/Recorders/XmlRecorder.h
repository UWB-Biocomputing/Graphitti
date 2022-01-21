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

#include "IRecorder.h"
#include "Model.h"

class XmlRecorder : public IRecorder {
	public:
		/// constructor
		XmlRecorder();

		/// destructor
		~XmlRecorder() override;

		static IRecorder* Create() { return new XmlRecorder(); }

		/// Initialize data
		/// Create a new xml file.
		void init() override;

		/// Init radii and rates history matrices with default values
		void initDefaultValues() override;

		/// Init radii and rates history matrices with current radii and rates
		void initValues() override;

		/// Get the current radii and rates vlaues
		void getValues() override;

		/// Terminate process
		void term() override;

		/// Compile history information in every epoch
		/// @param[in] vertices   The entire list of vertices.
		void compileHistories(AllVertices& vertices) override;

		/// Writes simulation results to an output destination.
		/// @param  vertices the Vertex list to search from.
		void saveSimData(const AllVertices& vertices) override;

		///  Prints out all parameters to logging file.
		///  Registered to OperationManager as Operation::printParameters
		void printParameters() override;

	protected:
		// a file stream for xml output
		std::ofstream resultOut_;

		// burstiness Histogram goes through the
		VectorMatrix burstinessHist_;

		// spikes history - history of accumulated spikes count of all neurons (10 ms bin)
		VectorMatrix spikesHistory_;

		// TODO: There seems to be multiple copies of this in different classes...
		void getStarterNeuronMatrix(VectorMatrix& matrix, const bool* starterMap);

};
