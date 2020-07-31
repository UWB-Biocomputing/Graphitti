/**
 * @file SynapseIndexMapTests.cpp
 *
 * @brief This file contains the unit tests for SynpaseIndexMap using GTest.
 *
 * @ingroup Testing
 */

#include "gtest/gtest.h"

#include "Simulation/Core/SynapseIndexMap.h"

/// Static variable used for neuron count initialization of the SynapseIndexMap in the tests
static int NEURON_COUNT = 50;

/// Static variable used for synapse count initialization of the SynapseIndexMap in the tests
static int SYNAPSE_COUNT = 100;

/// Testing object with initialized SynapseIndexMap. Used to reduce reused code.
struct SynapseIndexMapTestObject : public testing::Test {
    SynapseIndexMapTestObject() {
        synapseIndexMap = new SynapseIndexMap(NEURON_COUNT, SYNAPSE_COUNT);
    }

    SynapseIndexMap *synapseIndexMap;
};

TEST(SynapseIndexMap, DefaultConstructor) {
    SynapseIndexMap *synapseIndexMap = new SynapseIndexMap();
    EXPECT_TRUE(synapseIndexMap != nullptr);
}

TEST(SynapseIndexMap, OverloadedConstructor) {
    SynapseIndexMap *synapseIndexMap = new SynapseIndexMap(0, 0);
    EXPECT_TRUE(synapseIndexMap != nullptr);
}

TEST_F(SynapseIndexMapTestObject, OutgoingSynapseMapInitialiedSuccessfully) {
    EXPECT_EQ(synapseIndexMap->outgoingSynapseIndexMap_.size(), SYNAPSE_COUNT);
}

TEST_F(SynapseIndexMapTestObject, OutgoingSynapseBeginInitialiedSuccessfully) {
    EXPECT_EQ(synapseIndexMap->outgoingSynapseBegin_.size(), NEURON_COUNT);
}

TEST_F(SynapseIndexMapTestObject, OutgoingSynapseCountInitialiedSuccessfully) {
    EXPECT_EQ(synapseIndexMap->outgoingSynapseCount_.size(), NEURON_COUNT);
}

TEST_F(SynapseIndexMapTestObject, IncomingSynapseIndexMapInitialiedSuccessfully) {
    EXPECT_EQ(synapseIndexMap->incomingSynapseIndexMap_.size(), SYNAPSE_COUNT);
}

TEST_F(SynapseIndexMapTestObject, IncomingSynapseBeginInitialiedSuccessfully) {
    EXPECT_EQ(synapseIndexMap->incomingSynapseBegin_.size(), NEURON_COUNT);
}

TEST_F(SynapseIndexMapTestObject, IncomingSynapseCountInitializedSuccessfully) {
    EXPECT_EQ(synapseIndexMap->incomingSynapseCount_.size(), NEURON_COUNT);
}

