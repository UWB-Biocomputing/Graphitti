/**
 * @file GraphManagerTests.cpp
 * 
 * @brief This file contains unit tests for the GraphManager class
 * 
 * @ingroup Testing/UnitTesting
 */

#include "Global.h"
#include "GraphManager.h"
#include "gtest/gtest.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphml.hpp>

using namespace std;
string graphFile = "../configfiles/graphs/test-small-911.graphml";
string emptyGraphFile = "../configfiles/graphs/empty-graph.graphml";

TEST(GraphManager, GetInstanceReturnsInstance)
{
   GraphManager<NG911VertexProperty> *graphManager
      = &GraphManager<NG911VertexProperty>::getInstance();
   ASSERT_NE(graphManager, nullptr);
}

TEST(GraphManager, ReadGraphReturnsTrue)
{
   GraphManager<NG911VertexProperty> &graphManager
      = GraphManager<NG911VertexProperty>::getInstance();
   graphManager.setFilePath(graphFile);

   graphManager.registerProperty("objectID", &NG911VertexProperty::objectID);
   graphManager.registerProperty("name", &NG911VertexProperty::name);
   graphManager.registerProperty("type", &NG911VertexProperty::type);
   graphManager.registerProperty("y", &NG911VertexProperty::y);
   graphManager.registerProperty("x", &NG911VertexProperty::x);
   graphManager.registerProperty("servers", &NG911VertexProperty::servers);
   graphManager.registerProperty("trunks", &NG911VertexProperty::trunks);
   graphManager.registerProperty("segments", &NG911VertexProperty::segments);

   ASSERT_TRUE(graphManager.readGraph());
}

TEST(GraphManager, NumVerticesReturnsEleven)
{
   ASSERT_EQ(GraphManager<NG911VertexProperty>::getInstance().numVertices(), 11);
}

TEST(GraphManager, NumEdgesReturnsTwenty)
{
   ASSERT_EQ(GraphManager<NG911VertexProperty>::getInstance().numEdges(), 20);
}

TEST(GraphManager, GetVertcies)
{
   GraphManager<NG911VertexProperty>::VertexIterator vi, vi_end;
   GraphManager<NG911VertexProperty> &gm = GraphManager<NG911VertexProperty>::getInstance();
   boost::tie(vi, vi_end) = gm.vertices();
   ASSERT_EQ(*vi, 0);
   ASSERT_EQ(*vi_end, 11);
}

TEST(GraphManager, GetEdgesAndSource)
{
   GraphManager<NG911VertexProperty>::EdgeIterator ei, ei_end;
   GraphManager<NG911VertexProperty> &gm = GraphManager<NG911VertexProperty>::getInstance();

   list<GraphManager<NG911VertexProperty>::EdgeDescriptor> ei_list;
   for (boost::tie(ei, ei_end) = gm.edges(); ei != ei_end; ++ei) {
      ei_list.push_back(*ei);
   }
   auto start = ei_list.begin();
   ASSERT_EQ(gm.source(*start), 0);
}

TEST(GraphManager, GetEdgesAndTarget)
{
   GraphManager<NG911VertexProperty>::EdgeIterator ei, ei_end;
   GraphManager<NG911VertexProperty> &gm = GraphManager<NG911VertexProperty>::getInstance();

   list<GraphManager<NG911VertexProperty>::EdgeDescriptor> ei_list;
   for (boost::tie(ei, ei_end) = gm.edges(); ei != ei_end; ++ei) {
      ei_list.push_back(*ei);
   }
   auto start = ei_list.begin();
   ASSERT_EQ(gm.target(*start), 1);
}

TEST(GraphManager, SortEdges)
{
   GraphManager<NG911VertexProperty> &gm = GraphManager<NG911VertexProperty>::getInstance();
   auto list = gm.edgesSortByTarget();
   auto start = list.begin();
   ASSERT_EQ(gm.source(*start), 1);
   ASSERT_EQ(gm.target(*start), 0);
}

// These readGraph tests are after the Vertices/Edges tests so that
// the GraphManager singleton is not overwritten with the new graphs.
TEST(GraphManager, ReadEmptyGraph)
{
   GraphManager<NG911VertexProperty> &graphManager
      = GraphManager<NG911VertexProperty>::getInstance();
   graphManager.setFilePath(emptyGraphFile);
   ASSERT_TRUE(graphManager.readGraph());
}

TEST(GraphManager, ReadNonExistentGraph)
{
   GraphManager<NG911VertexProperty> &graphManager
      = GraphManager<NG911VertexProperty>::getInstance();
   graphManager.setFilePath("nonExistent.graphml");
   ASSERT_FALSE(graphManager.readGraph());
}
