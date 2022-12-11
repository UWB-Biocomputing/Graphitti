/**
 * @file SerializationTests.cpp
 *
 * @brief This file contains the unit tests for Serialization using GTest.
 *
 * @ingroup Testing/Core
 */

#include "Core.h"
#include "gtest/gtest.h"

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
};
