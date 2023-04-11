/**
 * @file EdgesFactoryTests.cpp
 *
 * @brief This file contains unit tests for the EdgesFactory using GTest.
 * 
 * @ingroup Testing/Core
 * 
 * We test that the EdgesFactory returns an instance of the correct class
 * we are requesting.
 */

#include "All911Edges.h"
#include "AllDSSynapses.h"
#include "AllDynamicSTDPSynapses.h"
#include "AllSTDPSynapses.h"
#include "AllSpikingSynapses.h"
#include "EdgesFactory.h"
#include "gtest/gtest.h"

TEST(EdgesFactory, GetInstanceReturnsInstance)
{
   EdgesFactory *edgesFactory = &EdgesFactory::getInstance();
   ASSERT_NE(nullptr, edgesFactory);
}

TEST(EdgesFactory, CreateAll911EdgesInstance)
{
   unique_ptr<AllEdges> edges = EdgesFactory::getInstance().createEdges("All911Edges");
   ASSERT_NE(nullptr, edges);
   ASSERT_NE(nullptr, dynamic_cast<All911Edges *>(edges.get()));
}

TEST(EdgesFactory, CreateAllDSSynapsesInstance)
{
   unique_ptr<AllEdges> edges = EdgesFactory::getInstance().createEdges("AllDSSynapses");
   ASSERT_NE(nullptr, edges);
   ASSERT_NE(nullptr, dynamic_cast<AllDSSynapses *>(edges.get()));
}

TEST(EdgesFactory, CreateAllDynamicSTDPSynapsesInstance)
{
   unique_ptr<AllEdges> edges = EdgesFactory::getInstance().createEdges("AllDynamicSTDPSynapses");
   ASSERT_NE(nullptr, edges);
   ASSERT_NE(nullptr, dynamic_cast<AllDynamicSTDPSynapses *>(edges.get()));
}

TEST(EdgesFactory, CreateAllSTDPSynapsesInstance)
{
   unique_ptr<AllEdges> edges = EdgesFactory::getInstance().createEdges("AllSTDPSynapses");
   ASSERT_NE(nullptr, edges);
   ASSERT_NE(nullptr, dynamic_cast<AllSTDPSynapses *>(edges.get()));
}

TEST(EdgesFactory, CreateAllSpikingSynapsesInstance)
{
   unique_ptr<AllEdges> edges = EdgesFactory::getInstance().createEdges("AllSpikingSynapses");
   ASSERT_NE(nullptr, edges);
   ASSERT_NE(nullptr, dynamic_cast<AllSpikingSynapses *>(edges.get()));
}

TEST(EdgesFactory, CreateNonExistentClassReturnsNullPtr)
{
   unique_ptr<AllEdges> edges = EdgesFactory::getInstance().createEdges("NonExistent");
   ASSERT_EQ(nullptr, edges);
}