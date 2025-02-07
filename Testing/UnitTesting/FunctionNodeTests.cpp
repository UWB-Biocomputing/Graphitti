/**
 * @file FunctionNodeTests.cpp
 *
 * @brief This class is used for unit testing the all FunctionNodes that inherit from IFunctionNode using GTest.
 *
 * @ingroup Testing/UnitTesting
 */

#include "GenericFunctionNode.h"
#include "OperationManagerTestingClass.h"
#include "gtest/gtest.h"

using namespace std;

/// Generic Function Node Tests
TEST(GenericFunctionNode, TemplateFunctionTest)
{
   Foo foo;
   function<void()> func = std::bind(&Foo::loadParameters, foo);
   IFunctionNode *chainNode = new GenericFunctionNode(Operations::loadParameters, func);
   ASSERT_TRUE(chainNode->invokeFunction(Operations::loadParameters));
}