//
// Created by Chris O'Keefe on 6/23/2020.
//

#include <iostream>
#include <Foo.h>
#include "ChainNode.h"
#include "gtest/gtest.h"

using namespace std;
using namespace std::placeholders;

TEST(ChainNode, TemplateFunctionTest) {
    Foo foo;
    function<void()> func = std::bind(&Foo::Burr, foo);
    IChainNode *chainNode = new ChainNode(func);
    chainNode->performOperation();
}