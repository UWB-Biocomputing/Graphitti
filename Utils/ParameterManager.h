/**
 * A class which contains and manages access to the XML 
 * parameter file used by a simulator instance at runtime.
 *
 * The class provides a simple interface to access 
 * parameters with the following assumptions:
 *   - The class' ::ReadParameters() method names the 
       expected Xpath for its own parameters.
 *   - The class makes all its own schema calls as needed.
 *   - The class will validate its own parameters unless 
 *     otherwise defined here.
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
#include <tinyxml.h>
#include <tinystr.h>

#include "BGTypes.h"

using namespace std;

class ParameterManager {
public:
   /// Get Instance method that returns a reference to this object.
   static ParameterManager &getInstance();

   bool loadParameterFile(string path);

   /// Interface methods for simulator objects
   bool getStringByXpath(string xpath, string &result);

   bool getIntByXpath(string xpath, int &var);

   bool getDoubleByXpath(string xpath, double &var);

   bool getFloatByXpath(string xpath, float &var);

   bool getBGFloatByXpath(string xpath, BGFLOAT &var);

   bool getLongByXpath(string xpath, long &var);

   /// Delete these methods because they can cause copy instances of the singleton when using threads.
   ParameterManager(ParameterManager const &) = delete;
   void operator=(ParameterManager const &) = delete;

private:
   /// Constructor is private to keep a singleton instance of this class.
   ParameterManager() {};

   TiXmlDocument *xmlDoc;
   TiXmlElement *root;

//   unique_ptr<TiXmlDocument> xmlDoc;
//   unique_ptr<TiXmlElement> root;

   bool checkDocumentStatus();
};
