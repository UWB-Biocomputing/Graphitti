/**
 * @file SerializationTests.cpp
 *
 * @brief This file contains the unit tests for Serialization using GTest.
 *
 * @ingroup Testing/Core
 */

#include "Core.h"
#include "gtest/gtest.h"
#include <iostream>
#include <tinyxml.h>

using namespace std;


// Run the simulator to generate a serialized file and check if it exist
TEST(Serialization, SerializeFileTest)
{
   // Test to check if Serialization is a success
   string executable = "./cgraphitti";
   string serialFileName = "../build/Output/serial1.xml";
   string configFile = "../configfiles/test-small-connected.xml";
   string argument = "-c " + configFile + " -w " + serialFileName;
   Core core;
   ASSERT_EQ(0, core.runSimulation(executable, argument));

   // Test to see if serialized file exist
   FILE *f = fopen("../build/Output/serial1.xml", "r");
   bool fileExist = f != NULL;
   fclose(f);
   ASSERT_TRUE(fileExist);

   // Test to confirm class version is correctly populated

   string classElements[] = {"AllEdges", "Connections"};
   string classVersion[] = {"1", "1"};
   int index = 0;

   // Load the XML file
   TiXmlDocument doc("../build/Output/serial1.xml");
   bool loadOkay = doc.LoadFile();

   // Check that the file was successfully loaded
   EXPECT_TRUE(loadOkay);

   if (loadOkay) {
      // Get the root element
      TiXmlElement *root = doc.RootElement();

      // Iterate through all the elements in the root
      for (TiXmlElement *elem = root->FirstChildElement(); elem != NULL;
           elem = elem->NextSiblingElement()) {
         // Get the element name
         string elemName = elem->Value();

         // Perform assertions on the element name
         EXPECT_EQ(elemName, classElements[index++]);

         //Iterate through all the attributes of the element
         for (TiXmlAttribute *attr = elem->FirstAttribute(); attr != NULL; attr = attr->Next()) {
            // Get the attribute name and value
            string attrName = attr->Name();
            string attrValue = attr->Value();

            // Perform assertions on the attribute name and value
            EXPECT_EQ(attrName, "cereal_class_version");
            EXPECT_EQ(attrValue, classVersion[index++]);
         }
      }
   }
};
