/**
 * @file DeserializationTests.cpp
 *
 * @brief This file contains the unit tests for Deserialization using GTest.
 *
 * @ingroup Testing/Core
 */

#include "Core.h"
#include "gtest/gtest.h"

using namespace std;

// Run check to confirm serialized file exist and then run the simulator to deserialize the file
TEST(Deserialization, DeserializeFileTest)
{
   string executable = "./cgraphitti";
   string serialFileName = "../build/Output/serial2.xml";   // FileName to write the serialized data
   string deserialFileName
      = "../build/Output/serial1.xml";   // FileName to read the serialized data
   string configFile = "../configfiles/test-small-connected.xml";

   // Test to see if serialized file exist to be deserialized
   FILE *f = fopen("../build/Output/serial1.xml", "r");
   bool fileExist = f != NULL;
   fclose(f);
   ASSERT_TRUE(fileExist);

   // Test to check if Deserialize the file is a success
   string argument = "-c " + configFile + " -w " + serialFileName + " -r " + deserialFileName;
   Core core;
   ASSERT_EQ(0, core.runSimulation(executable, argument));
};
