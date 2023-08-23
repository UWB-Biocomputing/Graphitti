
/**
 * @file SerializationFileAccessTest.cpp
 *
 * @brief This file contains the unit test to check if serialization file can 
 *        be accessed before the simulation starts, using GTest.
 *
 * @ingroup Testing/UnitTesting
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
   string nonWritableFilePath = "/root/non_writable_file.txt";
   string nonWritableArgument = "-c " + configFile + " -w " + nonWritableFilePath;

   Core core;

   // Test a non-writable file
   ASSERT_EQ(-1, core.runSimulation(executable, nonWritableArgument));
}

// Optional: create a similar test file to test a a positive case
// 1. Create a writable file
// string writableFilePath = "writable_file.txt";
// string writableArgument = "-c " + configFile + " -w " + writableFilePath;
// std::ofstream writableFile(writableFilePath);
// writableFile.close();
// 2. Test
// ASSERT_EQ(0, core.runSimulation(executable, writableArgument));
// 3. Clean up the writable file
// std::filesystem::remove(writableFilePath);