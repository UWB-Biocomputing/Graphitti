/**
 * @file SerializationTests.cpp
 *
 * @brief This file contains the unit tests for Serialization using GTest.
 *
 * @ingroup Testing/Core
 */

#include "Driver.h"
#include "gtest/gtest.h"

using namespace std;
string serialFileName = "../build/Output/serial1.xml";
string configFile = "../configfiles/test-small-connected.xml";

TEST(Serialization, SerializeFile)
{
   string argument = "-c " + configFile + " -w " + serialFileName;
   Driver driver;
   ASSERT_EQ(0, driver.setupSimulation(argument));
};

TEST(Serialization, CheckIfSerializedFileExist)
{
   FILE *f = fopen("../build/Output/serial1.xml", "r");
   bool fileExist = f != NULL;
   fclose(f);
   ASSERT_TRUE(fileExist);
};
