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

#include "Model.h"
#include "XmlRecorder.h"

class Xml911Recorder : public XmlRecorder {
	public:
		/// The constructor and destructor
		Xml911Recorder();

		~Xml911Recorder() override;

		static IRecorder* Create() { return new Xml911Recorder(); }

		/// Init radii and rates history matrices with default values
		void initDefaultValues() override;

		/// Init radii and rates history matrices with current radii and rates
		void initValues() override;

		/// Get the current radii and rates vlaues
		void getValues() override;

		/// Compile history information in every epoch
		///
		/// @param[in] vertices   The entire list of vertices.
		void compileHistories(AllVertices& vertices) override;

		/// Writes simulation results to an output destination.
		///
		/// @param  vertices the Vertex list to search from.
		void saveSimData(const AllVertices& vertices) override;

		///  Prints out all parameters to logging file.
		///  Registered to OperationManager as Operation::printParameters
		void printParameters() override;

	private:
};
