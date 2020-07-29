/**
 * @file SimulatorTests.cpp
 *
 * @brief This file contains the unit tests for Simulator using GTest.
 *
 * @ingroup Testing
 */

#include "gtest/gtest.h"

#include "Simulation/Core/Simulator.h"
#include <iostream>

using namespace std;

TEST(Simulator, GetInstanceSuccess) {
   Simulator *simulator = &Simulator::getInstance();
   ASSERT_TRUE(simulator != nullptr);
}

TEST(Simulator, OstreamSuccess) {
   ostream *os;
   EXPECT_NO_FATAL_FAILURE(Simulator::getInstance().printParameters(*os));
}

// advanceuntilgrowth
// freeresources