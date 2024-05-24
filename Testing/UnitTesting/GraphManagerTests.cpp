/**
 * @file GraphManagerTests.cpp
 * 
 * @brief This file contains unit tests for the GraphManager class
 * 
 * @ingroup Testing/UnitTesting
 */

#include "GraphManager.h"
#include "gtest/gtest.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphml.hpp>

using namespace std;
string graphFile = "../configfiles/graphs/test-small-911.graphml";
string emptyGraphFile = "../configfiles/graphs/empty-graph.graphml";

TEST(GraphManager, GetInstanceReturnsInstance)
{
   GraphManager *graphManager = &GraphManager::getInstance();
   ASSERT_NE(graphManager, nullptr);
}

TEST(GraphManager, ReadGraphReturnsTrue)
{
   GraphManager &graphManager = GraphManager::getInstance();
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

TEST(GraphManager, GetVertices)
{
   GraphManager::VertexIterator vi, vi_end;
   GraphManager &gm = GraphManager::getInstance();
   boost::tie(vi, vi_end) = gm.vertices();
   ASSERT_EQ(*vi, 0);
   ASSERT_EQ(*vi_end, 11);
}

TEST(GraphManager, GetEdgesAndSource)
{
   GraphManager::EdgeIterator ei, ei_end;
   GraphManager &gm = GraphManager::getInstance();
   boost::tie(ei, ei_end) = gm.edges();
   list<GraphManager::EdgeDescriptor> ei_list;
   for (boost::tie(ei, ei_end) = gm.edges(); ei != ei_end; ++ei) {
      ei_list.push_back(*ei);
   }
   auto start = ei_list.begin();
   ASSERT_EQ(gm.source(*start), 0);
}

TEST(GraphManager, GetEdgesAndTarget)
{
   GraphManager::EdgeIterator ei, ei_end;
   GraphManager &gm = GraphManager::getInstance();
   boost::tie(ei, ei_end) = gm.edges();
   list<GraphManager::EdgeDescriptor> ei_list;
   for (boost::tie(ei, ei_end) = gm.edges(); ei != ei_end; ++ei) {
      ei_list.push_back(*ei);
   }
   auto start = ei_list.begin();
   ASSERT_EQ(gm.target(*start), 1);
}

TEST(GraphManager, SortEdges)
{
   GraphManager &gm = GraphManager::getInstance();
   auto list = gm.edgesSortByTarget();
   auto start = list.begin();
   ASSERT_EQ(gm.source(*start), 1);
   ASSERT_EQ(gm.target(*start), 0);
}

// These readGraph tests are after the Verticies/Edges tests so that
// the GraphManager singleton is not overwritten with the new graphs.
TEST(GraphManager, ReadEmptyGraph)
{
   GraphManager &graphManager = GraphManager::getInstance();
   graphManager.setFilePath(emptyGraphFile);
   ASSERT_TRUE(graphManager.readGraph());
}

TEST(GraphManager, ReadNonExistentGraph)
{
   GraphManager &graphManager = GraphManager::getInstance();
   graphManager.setFilePath("nonExistent.graphml");
   ASSERT_FALSE(graphManager.readGraph());
}
