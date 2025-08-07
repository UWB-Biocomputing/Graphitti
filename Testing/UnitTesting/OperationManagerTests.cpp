/**
 * @file OperationManagerTests.cpp
 *
 * @brief  This class is used for unit testing the OperationManager using GTest.
 *
 * @ingroup Testing/UnitTesting
 *
 * These tests don't act independently since the OperationManager is a singleton. Each change to the state of
 * OperationManager will be carried through onto other tests.
 */

#include "Layout.h"
#include "OperationManager.h"
#include "OperationManagerTestingClass.h"
#include "Operations.h"
#include "gtest/gtest.h"

TEST(OperationManager, GetInstanceReturnsInstance)
{
   OperationManager *operationManager = &OperationManager::getInstance();
   ASSERT_TRUE(operationManager != nullptr);
}

TEST(OperationManager, AddingOneOperation)
{
   Foo foo;
   function<void()> function = std::bind(&Foo::loadParameters, foo);
   EXPECT_NO_FATAL_FAILURE(
      OperationManager::getInstance().registerOperation(Operations::loadParameters, function));
}

TEST(OperationManager, AddingManyOperations)
{
   Foo foo;
   function<void()> function = std::bind(&Foo::loadParameters, foo);
   for (int i = 0; i < 1000; i++) {
      EXPECT_NO_FATAL_FAILURE(
         OperationManager::getInstance().registerOperation(Operations::loadParameters, function));
   }
}

TEST(OperationManager, OperationExecutionSuccess)
{
   EXPECT_NO_FATAL_FAILURE(
      OperationManager::getInstance().executeOperation(Operations::loadParameters));
}

TEST(OperationManager, OperationExecutionContainsNoFunctionsOfOperationType)
{
   EXPECT_NO_FATAL_FAILURE(OperationManager::getInstance().executeOperation(Operations::copyToGPU));
}