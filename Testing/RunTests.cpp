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
    testing::InitGoogleTest();
    int result = RUN_ALL_TESTS();
    if (result != 0) {
        cerr << "\nError occured while running the tests. Error number: " << result << endl;
    }
    return 0;
}