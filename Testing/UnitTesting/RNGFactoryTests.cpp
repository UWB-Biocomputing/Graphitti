/**
 * @file RNGFactoryTests.cpp
 *
 * @brief This file contains unit tests for the RNGFactory using GTest.
 * 
 * @ingroup Testing/UnitTesting
 * 
 * We test that the RNGFactory returns an instance of the correct class
 * we are requesting.
 */

#include "MTRand.h"
#include "Norm.h"
#include "Utils/Factory.h"
#include "gtest/gtest.h"
#include <memory>

TEST(RNGFactory, GetInstanceReturnsInstance)
{
   Factory<MTRand> *rngFactory = &Factory<MTRand>::getInstance();
   ASSERT_NE(nullptr, rngFactory);
}

TEST(RNGFactory, CreateMTRandInstance)
{
   std::unique_ptr<MTRand> rng = Factory<MTRand>::getInstance().createType("MTRand");
   ASSERT_NE(nullptr, rng);
   ASSERT_NE(nullptr, dynamic_cast<MTRand *>(rng.get()));
}

TEST(RNGFactory, CreateNormInstance)
{
   std::unique_ptr<MTRand> rng = Factory<MTRand>::getInstance().createType("Norm");
   ASSERT_NE(nullptr, rng);
   ASSERT_NE(nullptr, dynamic_cast<Norm *>(rng.get()));
}

TEST(RNGFactory, CreateNonExistentClassReturnsNullPtr)
{
   std::unique_ptr<MTRand> rng = Factory<MTRand>::getInstance().createType("NonExistent");
   ASSERT_EQ(nullptr, rng);
}