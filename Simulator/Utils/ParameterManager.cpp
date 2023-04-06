/**
 * @file ParameterManager.cpp
 * 
 * @ingroup Simulator/Utils
 * 
 * @brief A class which contains and manages access to the XML 
 * parameter file used by a simulator instance at runtime.
 *
 * The class provides a simple interface to access parameters 
 * in an XML file.
 * It operates with the following assumptions:
 *   - The client's ::loadParameters method knows the XML 
 *     layout of the parameter file.
 *   - The client makes all its own schema calls as needed.
 *   - The client will validate its own parameters unless 
 *     otherwise defined here.
 *
 * This class makes use of TinyXPath, an open-source utility 
 * which enables XPath parsing of a TinyXml document object.
 * See the documentation here: http://tinyxpath.sourceforge.net/doc/index.html
 *
 * Created by Lizzy Presland, 2019
 * Supervised by Dr. Michael Stiber, UW Bothell CSSE Division
 */

#include "ParameterManager.h"
#include "BGTypes.h"
#include "xpath_static.h"
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <tinystr.h>
#include <tinyxml.h>
#include <vector>

/******************************************
* UTILITY METHODS
******************************************/
///@{

/// Get Instance method that returns a reference to this object.
ParameterManager &ParameterManager::getInstance()
{
   static ParameterManager instance;
   return instance;
}

/// Loads the XML file into a TinyXML tree.
/// This is the starting point for all XPath retrieval calls
/// for the client classes to use.
bool ParameterManager::loadParameterFile(string path)
{
   // load the XML document
   xmlDocument_.reset(new TiXmlDocument(path.c_str()));
   if (!xmlDocument_->LoadFile()) {
      cerr << "Failed loading simulation parameter file " << path << ":"
           << "\n\t" << xmlDocument_->ErrorDesc() << endl;
      cerr << " error row: " << xmlDocument_->ErrorRow()
           << ", error col: " << xmlDocument_->ErrorCol() << endl;
      return false;
   }
   // assign the document root_ object
   root_ = xmlDocument_->RootElement();
   return true;
}

/// A utility method to ensure the XML document objects exist.
/// If they don't exist, an XPath can't be computed and the
/// interface methods should terminate early.
bool ParameterManager::checkDocumentStatus()
{
   return xmlDocument_ != nullptr && root_ != nullptr;
}

/******************************************
* INTERFACE METHODS
******************************************/

/// Interface method to pull a string object from the xml
/// schema. The calling object must know the xpath to retrieve
/// the value.
///
/// @param xpath The xpath for the desired string value in the XML file
/// @param referenceVar The variable to store the string result into
/// @return bool A T/F flag indicating whether the retrieval succeeded
bool ParameterManager::getStringByXpath(string xpath, string &referenceVar)
{
   if (!checkDocumentStatus())
      return false;

   // temp string holds evaluation of value at xpath
   string temp;
   // raise error if tinyxml cannot compute the xpath's value or returns empty
   if (!TinyXPath::o_xpath_string(root_, xpath.c_str(), temp) || temp == "") {
      cerr << "Failed loading simulation parameter for xpath " << xpath << endl;
      return false;
   }
   referenceVar = temp;
   return true;
}

/// Interface method to pull an integer value from the xml
/// schema. The calling object must know the xpath to retrieve
/// the value.
///
/// @param xpath The xpath for the desired int value in the XML file
/// @param referenceVar The variable to store the int result into
/// @return bool A T/F flag indicating whether the retrieval succeeded
bool ParameterManager::getIntByXpath(string xpath, int &referenceVar)
{
   if (!checkDocumentStatus())
      return false;
   string tmp;
   if (!getStringByXpath(xpath, tmp)) {
      return false;
   }
   // Workaround for standard value conversion functions.
   // stoi() will cast floats to ints.
   if (regex_match(tmp, regex("\\d+[.]\\d+(e[+-]?\\d+)?f?|\\d+[.]?\\d+(e[+-]?\\d+)?f"))) {
      cerr << "Parsed parameter is likely a float/double value. "
           << "Terminating integer cast. Value: " << tmp << endl;
      return false;
   } else if (regex_match(tmp, regex(".*[^\\def.]+.*"))) {
      cerr << "Parsed parameter is likely a string. "
           << "Terminating integer cast. Value: " << tmp << endl;
      return false;
   }
   try {
      referenceVar = stoi(tmp);
   } catch (invalid_argument &arg_exception) {
      cerr << "Parsed parameter could not be parsed as an integer. Value: " << tmp << endl;
      return false;
   } catch (out_of_range &range_exception) {
      cerr << "Parsed string parameter could not be converted to an integer. Value: " << tmp
           << endl;
      return false;
   }
   return true;
}

/// Interface method to pull a double value from the xml
/// schema. The calling object must know the xpath to retrieve
/// the value.
///
/// @param xpath The xpath for the desired double value in the XML file
/// @param referenceVar The variable to store the double result into
/// @return bool A T/F flag indicating whether the retrieval succeeded
bool ParameterManager::getDoubleByXpath(string xpath, double &referenceVar)
{
   if (!checkDocumentStatus())
      return false;
   string tmp;
   if (!getStringByXpath(xpath, tmp)) {
      return false;
   }
   if (regex_match(tmp, regex(".*[^\\def.+-]+.*"))) {
      cerr << "Parsed parameter is likely a string. "
           << "Terminating double conversion. Value: " << tmp << endl;
      return false;
   }
   try {
      referenceVar = stod(tmp);
   } catch (invalid_argument &arg_exception) {
      cerr << "Parsed parameter could not be parsed as a double. Value: " << tmp << endl;
      return false;
   } catch (out_of_range &range_exception) {
      cerr << "Parsed string parameter could not be converted to a double. Value: " << tmp << endl;
      return false;
   }
   return true;
}

/// Interface method to pull a float value from the xml
/// schema. The calling object must know the xpath to retrieve
/// the value.
///
/// @param xpath The xpath for the desired float value in the XML file
/// @param referenceVariable The variable to store the float result into
/// @return bool A T/F flag indicating whether the retrieval succeeded
bool ParameterManager::getFloatByXpath(string xpath, float &referenceVariable)
{
   if (!checkDocumentStatus())
      return false;
   string tmp;
   if (!getStringByXpath(xpath, tmp)) {
      return false;
   }
   if (regex_match(tmp, regex(".*[^\\def.+-]+.*"))) {
      cerr << "Parsed parameter is likely a string. "
           << "Terminating double conversion. Value: " << tmp << endl;
      return false;
   }
   try {
      referenceVariable = stof(tmp);
   } catch (invalid_argument &arg_exception) {
      cerr << "Parsed parameter could not be parsed as a float. Value: " << tmp << endl;
      return false;
   } catch (out_of_range &range_exception) {
      cerr << "Parsed string parameter could not be converted to a float. Value: " << tmp << endl;
      return false;
   }
   return true;
}

/// Interface method to pull a BGFLOAT value from the xml
/// schema. The calling object must know the xpath to retrieve
/// the value.
///
/// This method is a wrapper to run the correct calls based on
/// how BGFLOAT is defined for the simulator. (For multi-threaded
/// usage, floats are used due to register size on GPUs, and
/// double usage is available for single-threaded CPU instances.)
///
/// @param xpath The xpath for the desired float value in the XML file
/// @param referenceVar The variable to store the float result into
/// @return bool A T/F flag indicating whether the retrieval succeeded
bool ParameterManager::getBGFloatByXpath(string xpath, BGFLOAT &referenceVar)
{
#ifdef SINGLEPRECISION
   return getFloatByXpath(xpath, referenceVar);
#endif
#ifdef DOUBLEPRECISION
   return getDoubleByXpath(xpath, referenceVar);
#endif
   cerr << "Could not infer primitive type for BGFLOAT variable." << endl;
   return false;
}

/// Interface method to pull a long value from the xml
/// schema. The calling object must know the xpath to retrieve
/// the value.
///
/// @param xpath The xpath for the desired float value in the XML file
/// @param referenceVariable The variable to store the long result into
/// @return bool A T/F flag indicating whether the retrieval succeeded
bool ParameterManager::getLongByXpath(string xpath, long &referenceVar)
{
   if (!checkDocumentStatus())
      return false;
   string tmp;
   if (!getStringByXpath(xpath, tmp)) {
      return false;
   }
   if (!regex_match(tmp, regex("[\\d]+l?"))) {
      cerr << "Parsed parameter is not a valid long format. "
           << "Terminating long conversion. Value: " << tmp << endl;
      return false;
   }
   try {
      referenceVar = stol(tmp);
   } catch (invalid_argument &arg_exception) {
      cerr << "Parsed parameter could not be parsed as a long. Value: " << tmp << endl;
      return false;
   } catch (out_of_range &range_exception) {
      cerr << "Parsed string parameter could not be converted to a long. Value: " << tmp << endl;
      return false;
   }
   return true;
}

/// Interface method to pull a list of ints into a vector<int> from the xml
/// schema. The calling object must match the elementName that stores the list. The xml files
/// in the NLists directory name the element by which type of neuron lists is expected.
///
/// A - Active Neuron List
/// I - Inhibitory Neuron List
/// P - Probed Neuron List
///
/// @param path the path to the xml file to read from
/// @param elementName The name of the element that stores the list of ints
/// @param referenceVariable The reference variable to store the list of ints
/// @return bool A T/F flag indicating whether the retrieval succeeded
bool ParameterManager::getIntVectorByXpath(const string &path, const string &elementName,
                                           vector<int> &referenceVar)
{
   // Open file using a local XmlDocument object
   TiXmlDocument xmlDocument(path.c_str());
   if (!xmlDocument.LoadFile()) {
      cerr << "Failed to load " << path.c_str() << ":"
           << "\n\t" << xmlDocument.ErrorDesc() << endl;
      return false;
   }

   // Check file for matching element
   TiXmlNode *xmlNode = nullptr;
   if ((xmlNode = xmlDocument.FirstChildElement(elementName)) == nullptr) {
      cerr << "Could not find <" << elementName << "> in vertex list file " << path << endl;
      return false;
   }

   // Get list of ints as a string stream
   std::istringstream valueStream(xmlNode->ToElement()->GetText());

   // Parse integers out of the string and add them to a list
   int i;
   while (valueStream.good()) {
      valueStream >> i;
      referenceVar.push_back(i);
   }
   return true;
}

/// Load the file indicated by the given string path into the ifstream object.
///
/// @param path   the path to the XML element containing the file name
/// @param file   ifstream object were file will be loaded
/// @return       true if the retrieval was successful, false otherwise
bool ParameterManager::getFileByXpath(const string &path, ifstream &file)
{
   string file_name;
   if (!ParameterManager::getInstance().getStringByXpath(path, file_name)) {
      cerr << "Could not find XML path: " << path << ".\n";
      return false;
   };

   // open file
   file.open(file_name.c_str());
   if (!file.is_open()) {
      cerr << "Failed to open file: " << file_name << ".\n";
      return false;
   }

   return true;
}