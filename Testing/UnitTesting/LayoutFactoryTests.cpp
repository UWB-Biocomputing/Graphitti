/**
 * @file LayoutFactoryTests.cpp
 *
 * @brief This file contains unit tests for the LayoutFactory using GTest.
 * 
 * @ingroup Testing/UnitTesting
 * 
 * We test that the LayoutFactory returns an instance of the correct class
 * we are requesting.
 */

#include "LayoutNeuro.h"
#include "Layout911.h"
#include "Utils/Factory.h"
#include "gtest/gtest.h"

TEST(LayoutFactory, GetInstanceReturnsInstance)
{
   Factory<Layout> *layoutFactory = &Factory<Layout>::getInstance();
   ASSERT_NE(nullptr, layoutFactory);
}


TEST(LayoutFactory, CreateLayoutNeuroInstance)
{
   unique_ptr<Layout> layout = Factory<Layout>::getInstance().createType("LayoutNeuro");
   ASSERT_NE(nullptr, layout);
   ASSERT_NE(nullptr, dynamic_cast<LayoutNeuro *>(layout.get()));
}

TEST(LayoutFactory, CreateLayout911Instance)
{
   unique_ptr<Layout> layout = Factory<Layout>::getInstance().createType("Layout911");
   ASSERT_NE(nullptr, layout);
   ASSERT_NE(nullptr, dynamic_cast<Layout911 *>(layout.get()));
}

TEST(LayoutFactory, CreateNonExistentClassReturnsNullPtr)
{
   unique_ptr<Layout> layout = Factory<Layout>::getInstance().createType("NonExistent");
   ASSERT_EQ(nullptr, layout);
}
