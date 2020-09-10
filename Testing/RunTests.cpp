/*
 * @file RunTests.cpp
 *
 * @brief This file is used to initiate Gtest and to run all tests in the project.
 *
 * @ingroup Testing
 */

#include <iostream>
#include <fstream>

#include "gtest/gtest.h"

#include <log4cplus/logger.h>
#include <log4cplus/configurator.h>
#include "log4cplus/loggingmacros.h"

using namespace std;

int main() {
   // Clear logging file at the start of testing
   fstream("Output/Debug/logging.txt", ios::out | ios::trunc);

   // Initialize log4cplus and set properties based on configure file
   ::log4cplus::initialize();
   ::log4cplus::PropertyConfigurator::doConfigure("RuntimeFiles/log4cplus_configure.ini");

   // Get the instance of the main logger and begin tests
   log4cplus::Logger logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("main"));
   LOG4CPLUS_INFO(logger, "Running Tests");

   // Disabling cout and cerr so output and error messages won't interrupt test flow.
   std::cout.setstate(std::ios_base::failbit);
   std::cerr.setstate(std::ios_base::failbit);

   // Initialize Google test and run all tests.
   testing::InitGoogleTest();
   return RUN_ALL_TESTS();
}