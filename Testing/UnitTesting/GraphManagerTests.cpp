/**
* @file GraphManagerTests.cpp
* 
* @brief This file contains unit tests for the GraphManager class
* 
* @ingroup Testing/UnitTesting
*/

#include "GraphManager.h"
#include "ParameterManager.h"
#include "gtest/gtest.h"

    

using namespace std;
string graphFile = "../configfiles/graphs/test-small-911.graphml";

string emptyGraphFile = "../configfiles/graphs/empty-graph.graphml";

TEST(GraphManager, GetInstanceReturnsInstance)
{
    GraphManager* graphManager = &GraphManager::getInstance();
    ASSERT_NE(graphManager, nullptr);
}

TEST(GraphManager, ReadGraphReturnsTrue)
{
    
    GraphManager& graphManager = GraphManager::getInstance();

    graphManager.setFilePath(graphFile);

    graphManager.registerProperty("objectID", &VertexProperty::objectID);
    graphManager.registerProperty("name", &VertexProperty::name);
    graphManager.registerProperty("type", &VertexProperty::type);
    graphManager.registerProperty("y", &VertexProperty::y);
    graphManager.registerProperty("x", &VertexProperty::x);
    graphManager.registerProperty("servers", &VertexProperty::servers);
    graphManager.registerProperty("trunks", &VertexProperty::trunks);
    graphManager.registerProperty("segments", &VertexProperty::segments);

    ASSERT_TRUE(graphManager.readGraph());
}

TEST(GraphManager, NumVerticiesReturnsEleven)
{
    ASSERT_EQ(GraphManager::getInstance().numVertices(), 11);
}

TEST(GraphManager, NumEdgesReturnsTwenty)
{
    ASSERT_EQ(GraphManager::getInstance().numEdges(), 20);
}

// These readGraph tests are after the numVerticies/Edges tests so that
// the GraphManager singleton is not overwritten with the new graphs.

TEST(GraphManager, ReadEmptyGraph)
{
    GraphManager& graphManager = GraphManager::getInstance();

    graphManager.setFilePath(emptyGraphFile);

    ASSERT_TRUE(graphManager.readGraph());
}

TEST(GraphManager, ReadNonExistentGraph)
{
    GraphManager& graphManager = GraphManager::getInstance();

    graphManager.setFilePath("nonExistent.graphml");

    ASSERT_FALSE(graphManager.readGraph());
}