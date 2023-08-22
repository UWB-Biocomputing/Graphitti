/**
 * @file SimulatorTests.cpp
 *
 * @brief This file contains the unit tests for Simulator using GTest.
 *
 * @ingroup Testing/UnitTesting
 */

#include "ParameterManager.h"
#include "Simulator.h"
#include "gtest/gtest.h"
#include <iostream>

using namespace std;

TEST(Simulator, GetInstanceSuccess)
{
   Simulator *simulator = &Simulator::getInstance();
   ASSERT_TRUE(simulator != nullptr);
}

TEST(Simulator, PrintParameters)
{
   EXPECT_NO_FATAL_FAILURE(Simulator::getInstance().printParameters());
}

TEST(Simulator, ParametersInitializedSuccessfully)
{
   ParameterManager::getInstance().loadParameterFile("../configfiles/test-medium-500.xml");
   Simulator::getInstance().loadParameters();

   EXPECT_EQ(30, Simulator::getInstance().getWidth());
   EXPECT_EQ(30, Simulator::getInstance().getHeight());
   EXPECT_EQ(BGFLOAT(100), Simulator::getInstance().getEpochDuration());
   EXPECT_EQ(500, Simulator::getInstance().getNumEpochs());
   EXPECT_EQ(200, Simulator::getInstance().getMaxFiringRate());
   EXPECT_EQ(200, Simulator::getInstance().getMaxEdgesPerVertex());
   EXPECT_EQ(1, Simulator::getInstance().getNoiseRngSeed());
   EXPECT_EQ(1, Simulator::getInstance().getInitRngSeed());
}


// advanceuntilgrowth
// freeresources