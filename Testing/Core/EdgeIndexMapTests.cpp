/**
 * @file SynapseIndexMapTests.cpp
 *
 * @brief This file contains the unit tests for SynpaseIndexMap using GTest.
 *
 * @ingroup Testing/Core
 */

#include "gtest/gtest.h"

#include "EdgeIndexMap.h"

/// Static variable used for vertex count initialization of the EdgeIndexMap in the tests
static int VERTEX_COUNT = 50;

/// Static variable used for synapse count initialization of the EdgeIndexMap in the tests
static int SYNAPSE_COUNT = 100;

/// Testing object with initialized EdgeIndexMap. Used to reduce reused code.
struct SynapseIndexMapTestObject : public testing::Test {
    SynapseIndexMapTestObject() {
        edgeIndexMap = new EdgeIndexMap(VERTEX_COUNT, SYNAPSE_COUNT);
    }

    EdgeIndexMap *edgeIndexMap;
};

TEST(EdgeIndexMap, DefaultConstructor) {
    EdgeIndexMap *edgeIndexMap = new EdgeIndexMap();
    EXPECT_TRUE(edgeIndexMap != nullptr);
}

TEST(EdgeIndexMap, OverloadedConstructor) {
    EdgeIndexMap *edgeIndexMap = new EdgeIndexMap(0, 0);
    EXPECT_TRUE(edgeIndexMap != nullptr);
}

// Tests are unavailable until member variables are switched from arrays to vectors.
// No way to check the size of an array that is allocated dynamically.
TEST_F(SynapseIndexMapTestObject, OutgoingSynapseMapInitialiedSuccessfully) {
    //EXPECT_EQ(sizeof(edgeIndexMap->outgoingSynapseIndexMap_), SYNAPSE_COUNT);
}

TEST_F(SynapseIndexMapTestObject, OutgoingSynapseBeginInitialiedSuccessfully) {
    //EXPECT_EQ(sizeof(edgeIndexMap->outgoingSynapseBegin_) / sizeof(edgeIndexMap->outgoingSynapseBegin_[0]) , VERTEX_COUNT);
}

TEST_F(SynapseIndexMapTestObject, OutgoingSynapseCountInitialiedSuccessfully) {
    //EXPECT_EQ(sizeof(edgeIndexMap->outgoingSynapseCount_) / sizeof(edgeIndexMap->outgoingSynapseCount_[0]), VERTEX_COUNT);
}

TEST_F(SynapseIndexMapTestObject, IncomingSynapseIndexMapInitialiedSuccessfully) {
    //EXPECT_EQ(sizeof(edgeIndexMap->incomingSynapseIndexMap_) / sizeof(edgeIndexMap->incomingSynapseIndexMap_[0]), SYNAPSE_COUNT);
}

TEST_F(SynapseIndexMapTestObject, IncomingSynapseBeginInitialiedSuccessfully) {
    //EXPECT_EQ(sizeof(edgeIndexMap->incomingSynapseBegin_) / sizeof(edgeIndexMap->incomingSynapseBegin_[0]), VERTEX_COUNT);
}

TEST_F(SynapseIndexMapTestObject, IncomingSynapseCountInitializedSuccessfully) {
    //EXPECT_EQ(sizeof(edgeIndexMap->incomingSynapseCount_) / sizeof(edgeIndexMap->incomingSynapseCount_[0]), VERTEX_COUNT);
}

