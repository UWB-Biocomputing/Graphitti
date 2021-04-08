/**
 * @file EdgeIndexMapTests.cpp
 *
 * @brief This file contains the unit tests for EdgeIndexMap using GTest.
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
    
    ~SynapseIndexMapTestObject() {
       delete edgeIndexMap;
    }


    EdgeIndexMap *edgeIndexMap;
};

TEST(EdgeIndexMap, DefaultConstructor) {
    EdgeIndexMap *edgeIndexMap = new EdgeIndexMap();
    EXPECT_TRUE(edgeIndexMap != nullptr);
    delete edgeIndexMap;
}

TEST(EdgeIndexMap, OverloadedConstructor) {
    EdgeIndexMap *edgeIndexMap = new EdgeIndexMap(VERTEX_COUNT, SYNAPSE_COUNT);
    EXPECT_TRUE(edgeIndexMap != nullptr);
    delete edgeIndexMap;
}

TEST(EdgeIndexMap, ZeroValueConstructor) {
   EdgeIndexMap *edgeIndexMap = new EdgeIndexMap(0, 0);
   EXPECT_TRUE(edgeIndexMap != nullptr);
   delete edgeIndexMap;
}

// Tests are unavailable until member variables are switched from arrays to vectors.
// No way to check the size of an array that is allocated dynamically.
TEST_F(SynapseIndexMapTestObject, OutgoingSynapseMapInitialiedSuccessfully) {
    //EXPECT_EQ(sizeof(edgeIndexMap->outgoingEdgeIndexMap_), SYNAPSE_COUNT);
}

TEST_F(SynapseIndexMapTestObject, OutgoingSynapseBeginInitialiedSuccessfully) {
    //EXPECT_EQ(sizeof(edgeIndexMap->outgoingEdgeBegin_) / sizeof(edgeIndexMap->outgoingEdgeBegin_[0]) , VERTEX_COUNT);
}

TEST_F(SynapseIndexMapTestObject, OutgoingSynapseCountInitialiedSuccessfully) {
    //EXPECT_EQ(sizeof(edgeIndexMap->outgoingEdgeCount_) / sizeof(edgeIndexMap->outgoingEdgeCount_[0]), VERTEX_COUNT);
}

TEST_F(SynapseIndexMapTestObject, IncomingSynapseIndexMapInitialiedSuccessfully) {
    //EXPECT_EQ(sizeof(edgeIndexMap->incomingEdgeIndexMap_) / sizeof(edgeIndexMap->incomingEdgeIndexMap_[0]), SYNAPSE_COUNT);
}

TEST_F(SynapseIndexMapTestObject, IncomingSynapseBeginInitialiedSuccessfully) {
    //EXPECT_EQ(sizeof(edgeIndexMap->incomingEdgeBegin_) / sizeof(edgeIndexMap->incomingEdgeBegin_[0]), VERTEX_COUNT);
}

TEST_F(SynapseIndexMapTestObject, IncomingSynapseCountInitializedSuccessfully) {
    //EXPECT_EQ(sizeof(edgeIndexMap->incomingEdgeCount_) / sizeof(edgeIndexMap->incomingEdgeCount_[0]), VERTEX_COUNT);
}

