/**
 * @file ParameterManager.h
 *
 * @brief A singleton class which contains and manages access to the XML
 * parameter file used by a simulator instance at runtime.
 *
 * @ingroup Simulator/Utils
 *
 * The class provides a simple interface to access 
 * parameters with the following assumptions:
 *   - The class' ::loadParameters() method names the 
 *     expected Xpath for its own parameters.
 *   - The class makes all its own schema calls as needed.
 *   - The class will validate its own parameters unless 
 *     otherwise defined here.
 *
 * This class supports multi-threaded programming.
 *
 * This class makes use of TinyXPath, an open-source utility 
 * which enables XPath parsing of a TinyXml document object.
 * See the documentation here: http://tinyxpath.sourceforge.net/doc/index.html
 *
 * Created by Lizzy Presland, 2019
 * Supervised by Dr. Michael Stiber, UW Bothell CSSE Division
 */

# pragma once

#include <memory>
#include <string>
#include <vector>

#include "BGTypes.h"
#include "tinyxml.h"

class ParameterManager {
	public:
		/// Get Instance method that returns a reference to this object.
		static ParameterManager& getInstance();

		/// Utility Methods
		~ParameterManager();

		bool loadParameterFile(std::string path);

		/// Interface methods for simulator objects
		bool getStringByXpath(std::string xpath, std::string& referenceVar);

		bool getIntByXpath(std::string xpath, int& referenceVar);

		bool getDoubleByXpath(std::string xpath, double& referenceVar);

		bool getFloatByXpath(std::string xpath, float& referenceVariable);

		bool getBGFloatByXpath(std::string xpath, BGFLOAT& referenceVar);

		bool getLongByXpath(std::string xpath, long& referenceVar);

		bool getIntVectorByXpath(const std::string& path, const std::string& elementName,
		                         std::vector<int>& referenceVar);

		/// Delete these methods because they can cause copy instances of the singleton when using threads.
		ParameterManager(const ParameterManager&) = delete;
		void operator=(const ParameterManager&) = delete;

	private:
		TiXmlDocument* xmlDocument_;
		TiXmlElement* root_;

		/// Constructor is private to keep a singleton instance of this class.
		ParameterManager();

		bool checkDocumentStatus();
};
