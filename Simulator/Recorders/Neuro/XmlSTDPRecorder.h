/**
 *      @file XmlSTDPRecorder.h
 *
 *       @ingroup Simulator/Recorders
 *
 *       @brief An implementation for recording spikes history in an XML file for spike timining dependent plasticity simulations
 */

/**
 ** \class XmlGrowthRecorder XmlGrowthRecorder.h "XmlGrowthRecorder.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The XmlGrowthRecorder provides a mechanism for recording neuron's layout, spikes history,
 ** and compile history information on xml file:
 **     -# neuron's locations, and type map,
 **     -# individual neuron's spike rate in epochs,
 **     -# network wide burstiness index data in 1s bins,
 **     -# network wide spike count in 10ms bins,
 **     -# individual neuron's radius history of every epoch.
 **
 ** \latexonly  \subsubsection*{Credits} \endlatexonly
 ** \htmlonly   <h3>Credits</h3> \endhtmlonly
 **
 ** Some models in this simulator is a rewrite of CSIM (2006) and other 
 ** work (Stiber and Kawasaki (2007?))
 **
 **
 **     @author Fumitaka Kawasaki, Snigdha Singh
 **/

#pragma once

#include <fstream>

#include "Model.h"
#include "XmlRecorder.h"

class XmlSTDPRecorder : public XmlRecorder {
	public:
		//! THe constructor and destructor
		XmlSTDPRecorder();

		~XmlSTDPRecorder() override;

		static IRecorder* Create() { return new XmlSTDPRecorder(); }

		/**
		 * Init radii and rates history matrices with default values
		 */
		void initDefaultValues() override;

		/**
		 * Init radii and rates history matrices with current radii and rates
		 */
		void initValues() override;

		/**
		 * Get the current radii and rates vlaues
		 */
		void getValues() override;

		/**
		 * Compile history information in every epoch
		 *
		 * @param[in] neurons   The entire list of neurons.
		 */
		void compileHistories(AllVertices& neurons) override;

		/**
		 * Writes simulation results to an output destination.
		 *
		 * @param  neurons the Neuron list to search from.
		 **/
		void saveSimData(const AllVertices& neurons) override;

		/**
		 *  Prints out all parameters to logging file.
		 *  Registered to OperationManager as Operation::printParameters
		 */
		void printParameters() override;

		virtual std::string toXML(std::string name, std::vector<std::vector<BGFLOAT>> MatrixToWrite) const;
		virtual std::string toXML(std::string name, std::vector<std::vector<int>> MatrixToWrite) const;

	protected:
		std::vector<std::vector<BGFLOAT>> weightsHistory_;
		std::vector<std::vector<int>> sourceNeuronIndexHistory_;
		std::vector<std::vector<int>> destNeuronIndexHistory_;

};
