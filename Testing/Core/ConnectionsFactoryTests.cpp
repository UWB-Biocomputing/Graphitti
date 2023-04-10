/**
 * @file ConnectionsFactoryTests.cpp
 *
 * @brief This file contains unit tests for the ConnectionsFactory using GTest.
 * 
 * @ingroup Testing/Core
 * 
 * We test that the ConnectionsFactory returns an instance of the correct class
 * we are requesting.
 */

#include "ConnGrowth.h"
#include "ConnStatic.h"
#include "Connections911.h"
#include "ConnectionsFactory.h"
#include "gtest/gtest.h"

TEST(ConnectionsFactory, GetInstanceReturnsInstance)
{
   ConnectionsFactory *connectionsFactory = &ConnectionsFactory::getInstance();
   ASSERT_NE(nullptr, connectionsFactory);
}

TEST(ConnectionsFactory, CreateConnstaticInstance)
{
   unique_ptr<Connections> connections
      = ConnectionsFactory::getInstance().createConnections("ConnStatic");
   ASSERT_NE(nullptr, connections);
   ASSERT_NE(nullptr, dynamic_cast<ConnStatic *>(connections.get()));
}

TEST(ConnectionsFactory, CreateConnGrowthInstance)
{
   unique_ptr<Connections> connections
      = ConnectionsFactory::getInstance().createConnections("ConnGrowth");
   ASSERT_NE(nullptr, connections);
   ASSERT_NE(nullptr, dynamic_cast<ConnGrowth *>(connections.get()));
}

TEST(ConnectionsFactory, CreateConnections911Instance)
{
   unique_ptr<Connections> connections
      = ConnectionsFactory::getInstance().createConnections("Connections911");
   ASSERT_NE(nullptr, connections);
   ASSERT_NE(nullptr, dynamic_cast<Connections911 *>(connections.get()));
}

TEST(ConnectionsFactory, CreateNonExistentClassReturnsNullPtr)
{
   unique_ptr<Connections> connections
      = ConnectionsFactory::getInstance().createConnections("NonExistent");
   ASSERT_EQ(nullptr, connections);
}