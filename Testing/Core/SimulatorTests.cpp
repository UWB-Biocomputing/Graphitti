/**
 * @file SimulatorTests.cpp
 *
 * @brief This file contains the unit tests for Simulator using GTest.
 *
 * @ingroup Testing
 */

#include "gtest/gtest.h"

#include "ParameterManager.h"
#include "Simulation/Core/Simulator.h"
#include <iostream>

using namespace std;

TEST(Simulator, GetInstanceSuccess) {
   Simulator *simulator = &Simulator::getInstance();
   ASSERT_TRUE(simulator != nullptr);
}

TEST(Simulator, PrintParameters) {
   EXPECT_NO_FATAL_FAILURE(Simulator::getInstance().printParameters());
}

TEST(Simulator, ParametersInitializedSuccessfully) {
   ParameterManager::getInstance().loadParameterFile("configfiles/test-medium-500.xml");
   Simulator::getInstance().loadParameters();

   EXPECT_EQ(30, Simulator::getInstance().getWidth());
   EXPECT_EQ(30, Simulator::getInstance().getHeight());
   EXPECT_EQ(BGFLOAT(100), Simulator::getInstance().getEpochDuration());
   EXPECT_EQ(500, Simulator::getInstance().getNumEpochs());
   EXPECT_EQ(200, Simulator::getInstance().getMaxFiringRate());
   EXPECT_EQ(200, Simulator::getInstance().getMaxSynapsesPerNeuron());
   EXPECT_EQ(1, Simulator::getInstance().getSeed());
   EXPECT_EQ("results/test-medium-500-out.xml", Simulator::getInstance().getResultFileName());
}


// advanceuntilgrowth
// freeresources