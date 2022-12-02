/**
 * @file DeserializationTests.cpp
 *
 * @brief This file contains the unit tests for Deserialization using GTest.
 *
 * @ingroup Testing/Core
 */

#include "Driver.h"
#include "gtest/gtest.h"

using namespace std;
string serialFileName = "../build/Output/serial2.xml";     // FileName to write the serialized data
string deserialFileName = "../build/Output/serial1.xml";   // FileName to read the serialized data
string configFile = "../configfiles/test-small-connected.xml";

TEST(Serialization, CheckIfSerializedFileExist)
{
   FILE *f = fopen("../build/Output/serial1.xml", "r");
   bool fileExist = f != NULL;
   fclose(f);
   ASSERT_TRUE(fileExist);
};

TEST(Serialization, DeserializeFile)
{
   string argument = "-c " + configFile + " -w " + serialFileName + " -r " + deserialFileName;
   Driver driver;
   ASSERT_EQ(0, driver.setupSimulation(argument));
};
