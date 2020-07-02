//
// Created by Chris O'Keefe on 6/26/2020.
//

#include "../../Core/OperationManager.h"
#include "GenericFunctionNode.h"
#include "IFunctionNode.h"
#include "Operations.h"
#include "Foo.h"
#include "gtest/gtest.h"

/**
 * This class is used for testing the OperationManager using GTest.
 *
 * These tests don't act independently since the OperationManager is a singleton. Each change to the state of
 * OperationManager will be carried through onto other tests. In other words, some tests rely on previous ones to
 * run properly. These tests should be executed in sequence.
 */


TEST(OperationManager, GetInstanceReturnsInstance) {
    ASSERT_TRUE(OperationManager::getInstance() != nullptr);
}

TEST(OperationManager, AddNodeToChain) {
    Foo foo;
    function<void()> function = std::bind(&Foo::deallocateMemory, foo);
    ASSERT_TRUE(OperationManager::getInstance()->registerOperation(Operations::op::deallocateMemory, function));
}

TEST(OperationManager, OperationPassedToChain) {
    ASSERT_TRUE(OperationManager::getInstance()->executeOperation(Operations::op::deallocateMemory));
}

TEST(OperationManagerTestObject, OperationPassedToEmptyChain) {
    ASSERT_FALSE(OperationManager::getInstance()->executeOperation(Operations::op::copyFromGPU));
}