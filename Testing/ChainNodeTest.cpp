//
// Created by chris on 6/23/2020.
//

#include <iostream>
#include "IChainNode.h"
#include "Dog.h"
#include "Fish.h"
#include "gtest/gtest.h"

using namespace std;

int main(int argc, char *argv[]) {
    testing::InitGoogleTest();
    RUN_ALL_TESTS();
    return 0;
}

struct ChainNodeTest : testing::Test {
    IChainNode *fish;
    IChainNode *dog1;
    IChainNode *dog2;

    ChainNodeTest() {
        fish = new Fish();
        dog1 = new Dog();
        dog2 = new Dog();

        dog1->SetNextNode(fish)->SetNextNode(dog2);
    }

    virtual  ~ChainNodeTest() {
        delete fish;
        delete dog1;
        delete dog2;
    }
};

TEST_F(ChainNodeTest, OperationSuccessOnFirstNode) {
    EXPECT_EQ("There's a dog in the chain", dog1->PerformOperation("dog"));
}

TEST_F(ChainNodeTest, OperationSuccessOnSecondNode) {
    EXPECT_EQ("There's a fish in the chain", dog1->PerformOperation("fish"));
}

TEST_F(ChainNodeTest, OperationSuccessOnLastNode) {
    EXPECT_EQ("There's a dog in the chain", fish->PerformOperation("dog"));
}

TEST_F(ChainNodeTest, OperationIncomplete) {
    EXPECT_EQ("Request can't be processed", dog1->PerformOperation("cat"));
}