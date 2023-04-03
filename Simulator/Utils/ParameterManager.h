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

#pragma once

#include "BGTypes.h"
#include "tinyxml.h"
#include <memory>
#include <string>
#include <vector>

using namespace std;

class ParameterManager {
public:
   /// Get Instance method that returns a reference to this object.
   static ParameterManager &getInstance();

   /// Utility Methods
   ~ParameterManager();

   bool loadParameterFile(string path);

   /// Interface methods for simulator objects
   bool getStringByXpath(string xpath, string &referenceVar);

   bool getIntByXpath(string xpath, int &referenceVar);

   bool getDoubleByXpath(string xpath, double &referenceVar);

   bool getFloatByXpath(string xpath, float &referenceVariable);

   bool getBGFloatByXpath(string xpath, BGFLOAT &referenceVar);

   bool getLongByXpath(string xpath, long &referenceVar);

   bool getIntVectorByXpath(const string &path, const string &elementName,
                            vector<int> &referenceVar);

   bool getFileByXpath(const string &path, ifstream &file);

   /// Delete copy and move methods to avoid copy instances of the singleton
   ParameterManager(const ParameterManager &parameterManager) = delete;
   ParameterManager &operator=(const ParameterManager &parameterManager) = delete;

   ParameterManager(ParameterManager &&parameterManager) = delete;
   ParameterManager &operator=(ParameterManager &&parameterManager) = delete;

private:
   unique_ptr<TiXmlDocument> xmlDocument_;
   TiXmlElement *root_;

   /// Constructor is private to keep a singleton instance of this class.
   ParameterManager();

   bool checkDocumentStatus();
};
