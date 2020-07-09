//
// Created by Chris O'Keefe on 6/23/2020.
//

#include <iostream>
#include <Foo.h>
#include "GenericFunctionNode.h"
#include "gtest/gtest.h"

/**
 * This class is used for testing the GenericFunctionNode using GTest.
 */


using namespace std;

TEST(ChainNode, TemplateFunctionTest) {
    Foo foo;
    function<void()> func = std::bind(&Foo::allocateMemory, foo);
    IFunctionNode *chainNode = new GenericFunctionNode(Operations::op::allocateMemory, func);
    ASSERT_TRUE(chainNode->invokeFunction(Operations::op::allocateMemory));
}