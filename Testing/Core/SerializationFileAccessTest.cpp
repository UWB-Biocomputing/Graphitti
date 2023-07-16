
/**
 * @file SerializationFileAccessTest.cpp
 *
 * @brief This file contains the unit test to check if serialization file can 
 *        be accessed before the simulation starts, using GTest.
 *
 * @ingroup Testing/Core
 */

#include "Core.h"
#include "gtest/gtest.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <tinyxml.h>

using namespace std;

// Test case for isSerializedFileWritable function
TEST(SerializationFileAccessTest, FileWritableTest)
{
   string executable = "./cgraphitti";
   string configFile = "../configfiles/test-small-connected.xml";

   string writableFilePath = "writable_file.txt";
   string nonWritableFilePath = "/root/non_writable_file.txt";

   string writableArgument = "-c " + configFile + " -w " + writableFilePath;
   string nonWritableArgument = "-c " + configFile + " -w " + nonWritableFilePath;

   // Create a writable file
   std::ofstream writableFile(writableFilePath);
   writableFile.close();

   Core core;
   // Test a writable file
   // ASSERT_EQ(0, core.runSimulation(executable, writableArgument));
   // Test a non-writable file
   ASSERT_EQ(-1, core.runSimulation(executable, nonWritableArgument));

   // Clean up the writable file
   std::filesystem::remove(writableFilePath);
}