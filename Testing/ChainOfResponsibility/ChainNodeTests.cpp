//
// Created by Chris O'Keefe on 6/23/2020.
//

#include <iostream>
#include <Foo.h>
#include "ChainOperationHandler.h"
#include "ChainNode.h"
#include "gtest/gtest.h"

using namespace std;
using namespace std::placeholders;

TEST(ChainNode, TemplateFunctionTest) {
    Foo foo;
    function<void()> func = std::bind(&Foo::Burr, foo);
    IChainNode *chainNode = new ChainNode<Foo>(foo, func);
    chainNode->performOperation();
}

//struct ChainNodeTest : testing::Test {
//    ChainNode *fish;
//    ChainNode *dog1;
//    ChainNode *dog2;
//
//    ChainNodeTest() {
//        fish = new Fish();
//        dog1 = new Dog();
//        dog2 = new Dog();
//
//        dog1->setNextNode(fish)->setNextNode(dog2);
//    }
//
//    virtual  ~ChainNodeTest() {
//        delete fish;
//        delete dog1;
//        delete dog2;
//    }
//};
//
//TEST_F(ChainNodeTest, OperationSuccessOnFirstNode) {
//    EXPECT_EQ("There's a dog in the chain", dog1->performOperation("dog"));
//}
//
//TEST_F(ChainNodeTest, OperationSuccessOnSecondNode) {
//    EXPECT_EQ("There's a fish in the chain", dog1->performOperation("fish"));
//}
//
//TEST_F(ChainNodeTest, OperationSuccessOnLastNode) {
//    EXPECT_EQ("There's a dog in the chain", fish->performOperation("dog"));
//}
//
//TEST_F(ChainNodeTest, OperationIncomplete) {
//    EXPECT_EQ("Request can't be processed", dog1->performOperation("cat"));
//}