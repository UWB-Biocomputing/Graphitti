//
// Created by Chris O'Keefe on 6/26/2020.
//

#include "ChainOperationManager.h"
#include "ChainNode.h"
#include "IChainNode.h"
#include "Operations.h"
#include "Foo.h"
#include "gtest/gtest.h"

struct ChainOperationManagerTest : public testing::Test {
    ChainOperationManagerTest() {
        Foo foo;
        function<void()> func = std::bind(&Foo::Burr, foo);
        IChainNode *chainNode = new ChainNode(func);
        ChainOperationManager::getInstance()->addNodeToChain(Operations::op::allocateMemory, chainNode);
    }
};


TEST(ChainOperationManager, GetInstanceReturnsInstance) {
    ASSERT_TRUE(ChainOperationManager::getInstance() != nullptr);
}

TEST_F(ChainOperationManagerTest, AddNodeToChain) {
    Foo foo1;
    function<void()> func1 = std::bind(&Foo::BurrSure, foo1);
    IChainNode *chainNode1 = new ChainNode(func1);
    ASSERT_TRUE(ChainOperationManager::getInstance()->addNodeToChain(Operations::op::allocateMemory, chainNode1));
}

TEST_F(ChainOperationManagerTest, OperationPassedToChain) {
    ChainOperationManager::getInstance()->executeOperation(Operations::op::allocateMemory);
    //ASSERT_TRUE(ChainOperationManager::getInstance()->executeOperation(Operations::op::allocateMemory));
}

TEST_F(ChainOperationManagerTest, OperationPassedToEmptyChain) {
    ASSERT_FALSE(ChainOperationManager::getInstance()->executeOperation(Operations::op::copyFromGPU));
}