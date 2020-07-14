#include "gtest/gtest.h"

#include "Foo.h"
#include "Core/OperationManager.h"
#include "Core/Operations.h"

/**
 * This class is used for testing the OperationManager using GTest.
 *
 * These tests don't act independently since the OperationManager is a singleton. Each change to the state of
 * OperationManager will be carried through onto other tests. In other words, some tests rely on previous ones to
 * run properly. These tests should be executed in sequence.
 */

TEST(OperationManager, AddingOneOperation) {
    Foo foo;
    function<void()> function = std::bind(&Foo::deallocateMemory, foo);
    OperationManager::getInstance().registerOperation(Operations::op::deallocateMemory, function);
}

TEST(OperationManager, AddingManyOperations) {
    Foo foo;
    function<void()> function = std::bind(&Foo::deallocateMemory, foo);
    for (int i = 0; i < 10000; i++) {
        OperationManager::getInstance().registerOperation(Operations::op::deallocateMemory, function);
    }
}

TEST(OperationManager, OperationExecutionSuccess) {
    OperationManager::getInstance().executeOperation(Operations::op::deallocateMemory);
}

TEST(OperationManager, OperationExecutionContainsNoFunctionsOfOperationType) {
    OperationManager::getInstance().executeOperation(Operations::copyToGPU);
}