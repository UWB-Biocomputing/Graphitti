/**
 * @file SynapseIndexMapTests.cpp
 *
 * @brief This file contains the unit tests for SynpaseIndexMap using GTest.
 *
 * @ingroup Testing/Core
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

// Tests are unavailable until member variables are switched from arrays to vectors.
// No way to check the size of an array that is allocated dynamically.
TEST_F(SynapseIndexMapTestObject, OutgoingSynapseMapInitialiedSuccessfully) {
    //EXPECT_EQ(sizeof(synapseIndexMap->outgoingSynapseIndexMap_), SYNAPSE_COUNT);
}

TEST_F(SynapseIndexMapTestObject, OutgoingSynapseBeginInitialiedSuccessfully) {
    //EXPECT_EQ(sizeof(synapseIndexMap->outgoingSynapseBegin_) / sizeof(synapseIndexMap->outgoingSynapseBegin_[0]) , NEURON_COUNT);
}

TEST_F(SynapseIndexMapTestObject, OutgoingSynapseCountInitialiedSuccessfully) {
    //EXPECT_EQ(sizeof(synapseIndexMap->outgoingSynapseCount_) / sizeof(synapseIndexMap->outgoingSynapseCount_[0]), NEURON_COUNT);
}

TEST_F(SynapseIndexMapTestObject, IncomingSynapseIndexMapInitialiedSuccessfully) {
    //EXPECT_EQ(sizeof(synapseIndexMap->incomingSynapseIndexMap_) / sizeof(synapseIndexMap->incomingSynapseIndexMap_[0]), SYNAPSE_COUNT);
}

TEST_F(SynapseIndexMapTestObject, IncomingSynapseBeginInitialiedSuccessfully) {
    //EXPECT_EQ(sizeof(synapseIndexMap->incomingSynapseBegin_) / sizeof(synapseIndexMap->incomingSynapseBegin_[0]), NEURON_COUNT);
}

TEST_F(SynapseIndexMapTestObject, IncomingSynapseCountInitializedSuccessfully) {
    //EXPECT_EQ(sizeof(synapseIndexMap->incomingSynapseCount_) / sizeof(synapseIndexMap->incomingSynapseCount_[0]), NEURON_COUNT);
}

