/*
 * @file RunTests.cpp
 *
 * @brief This file is used to initiate Gtest and to run all tests in the project.
 *
 * @ingroup Testing
 */

#include <iostream>

#include "gtest/gtest.h"

using namespace std;

int main() {
   // Disabling cout and cerr so output and error messages won't interrupt test flow.
   std::cout.setstate(std::ios_base::failbit);
   std::cerr.setstate(std::ios_base::failbit);

   // Initialize Google test and run all tests.
   testing::InitGoogleTest();
   int result = RUN_ALL_TESTS();

   // Check result of the tests.
   if (result != 0) {
      cerr << "\nError occured while running the tests. Error number: " << result << endl;
   }
   return 0;
}