/**
 * @file OperationManagerTests.cpp
 *
 * @brief  This class is used for unit testing the OperationManager using GTest.
 *
 * @ingroup Testing
 *
 * These tests don't act independently since the OperationManager is a singleton. Each change to the state of
 * OperationManager will be carried through onto other tests.
 */

#include "gtest/gtest.h"

#include "OperationManagerTestingClass.h"
#include "Simulation/Core/OperationManager.h"
#include "Simulation/Core/Operations.h"
#include "Layout.h"

TEST(OperationManager, GetInstanceReturnsInstance) {
   OperationManager *operationManager = &OperationManager::getInstance();
   ASSERT_TRUE(operationManager != nullptr);
}

TEST(OperationManager, AddingOneOperation) {
   Foo foo;
   function<void()> function = std::bind(&Foo::loadParameters, foo);
   EXPECT_NO_FATAL_FAILURE(
         OperationManager::getInstance().registerOperation(Operations::op::loadParameters, function));
}

TEST(OperationManager, AddingManyOperations) {
   Foo foo;
   function<void()> function = std::bind(&Foo::loadParameters, foo);
   for (int i = 0; i < 1000; i++) {
      EXPECT_NO_FATAL_FAILURE(
            OperationManager::getInstance().registerOperation(Operations::op::loadParameters, function));
   }
}

TEST(OperationManager, OperationExecutionSuccess) {
   EXPECT_NO_FATAL_FAILURE(OperationManager::getInstance().executeOperation(Operations::op::loadParameters));
}

TEST(OperationManager, OperationExecutionContainsNoFunctionsOfOperationType) {
   EXPECT_NO_FATAL_FAILURE(OperationManager::getInstance().executeOperation(Operations::op::copyToGPU));
}