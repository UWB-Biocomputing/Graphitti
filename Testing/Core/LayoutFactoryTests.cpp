/**
 * @file LayoutFactoryTests.cpp
 *
 * @brief This file contains unit tests for the LayoutFactory using GTest.
 * 
 * @ingroup Testing/Core
 * 
 * We test that the LayoutFactory returns an instance of the correct class
 * we are requesting.
 */

#include "DynamicLayout.h"
#include "FixedLayout.h"
#include "Layout911.h"
#include "LayoutFactory.h"
#include "gtest/gtest.h"

TEST(LayoutFactory, GetInstanceReturnsInstance)
{
   LayoutFactory *layoutFactory = &LayoutFactory::getInstance();
   ASSERT_NE(nullptr, layoutFactory);
}

TEST(LayoutFactory, CreateDynamicLayoutInstance)
{
   unique_ptr<Layout> layout = LayoutFactory::getInstance().createLayout("DynamicLayout");
   ASSERT_NE(nullptr, layout);
   ASSERT_NE(nullptr, dynamic_cast<DynamicLayout *>(layout.get()));
}

TEST(LayoutFactory, CreateFixedLayoutInstance)
{
   unique_ptr<Layout> layout = LayoutFactory::getInstance().createLayout("FixedLayout");
   ASSERT_NE(nullptr, layout);
   ASSERT_NE(nullptr, dynamic_cast<FixedLayout *>(layout.get()));
}

TEST(LayoutFactory, CreateLayout911Instance)
{
   unique_ptr<Layout> layout = LayoutFactory::getInstance().createLayout("Layout911");
   ASSERT_NE(nullptr, layout);
   ASSERT_NE(nullptr, dynamic_cast<Layout911 *>(layout.get()));
}

TEST(LayoutFactory, CreateNonExistentClassReturnsNullPtr)
{
   unique_ptr<Layout> layout = LayoutFactory::getInstance().createLayout("NonExistent");
   ASSERT_EQ(nullptr, layout);
}