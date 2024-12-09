/**
 * @file SerializationHelper.cpp
 *
 * @brief Helper class that contains utility functions for serialization testing.
 *  
 * @ingroup Testing/UnitTesting
 */

#include "Core.h"
#include <filesystem>
#include <fstream>
#include <tinyxml.h>

using namespace std;
namespace fs = std::filesystem;

// Helper function to run the simulation
bool runSimulation(const string &executable, const string &arguments)
{
   Core core;
   return core.runSimulation(executable, arguments) == 0 ? true : false;
}

// Helper function to check if a file exists
bool fileExists(const string &filePath)
{
   return fs::exists(filePath);
}

// Helper function to compare two XML elements recursively
bool compareXmlElements(TiXmlElement *elem1, TiXmlElement *elem2)
{
   if (!elem1 || !elem2) {
      return false;
   }

   // Compare element names
   if (std::string(elem1->Value()) != std::string(elem2->Value())) {
      return false;
   }

   // Compare attributes
   TiXmlAttribute *attr1 = elem1->FirstAttribute();
   TiXmlAttribute *attr2 = elem2->FirstAttribute();

   while (attr1 && attr2) {
      if (std::string(attr1->Name()) != std::string(attr2->Name())
          || std::string(attr1->Value()) != std::string(attr2->Value())) {
         return false;
      }
      attr1 = attr1->Next();
      attr2 = attr2->Next();
   }

   // Check if one element has more attributes than the other
   if (attr1 || attr2) {
      return false;
   }

   // Compare child elements
   TiXmlElement *child1 = elem1->FirstChildElement();
   TiXmlElement *child2 = elem2->FirstChildElement();

   while (child1 && child2) {
      if (!compareXmlElements(child1, child2)) {
         return false;
      }
      child1 = child1->NextSiblingElement();
      child2 = child2->NextSiblingElement();
   }

   // Check if one element has more children than the other
   if (child1 || child2) {
      return false;
   }

   return true;
}

// Helper function to compare XML files
bool compareXmlFiles(const std::string &filePath1, const std::string &filePath2)
{
   TiXmlDocument doc1, doc2;

   // Load the first XML file
   if (!doc1.LoadFile(filePath1.c_str())) {
      std::cerr << "Failed to load XML file: " << filePath1 << std::endl;
      return false;
   }

   // Load the second XML file
   if (!doc2.LoadFile(filePath2.c_str())) {
      std::cerr << "Failed to load XML file: " << filePath2 << std::endl;
      return false;
   }

   // Compare the root elements of both XML documents
   TiXmlElement *root1 = doc1.RootElement();
   TiXmlElement *root2 = doc2.RootElement();

   if (!root1 || !root2) {
      std::cerr << "One of the XML files has no root element." << std::endl;
      return false;
   }

   // Recursively compare the XML structures
   return compareXmlElements(root1, root2);
}