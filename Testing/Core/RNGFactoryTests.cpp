/**
 * @file RNGFactoryTests.cpp
 *
 * @brief This file contains unit tests for the RNGFactory using GTest.
 * 
 * @ingroup Testing/Core
 * 
 * We test that the RNGFactory returns an instance of the correct class
 * we are requesting.
 */

#include "MTRand.h"
#include "Norm.h"
#include "RNGFactory.h"
#include "gtest/gtest.h"

TEST(RNGFactory, GetInstanceReturnsInstance)
{
   RNGFactory *rngFactory = &RNGFactory::getInstance();
   ASSERT_NE(nullptr, rngFactory);
}

TEST(RNGFactory, CreateMTRandInstance)
{
   unique_ptr<MTRand> rng = RNGFactory::getInstance().createRNG("MTRand");
   ASSERT_NE(nullptr, rng);
   ASSERT_NE(nullptr, dynamic_cast<MTRand *>(rng.get()));
}

TEST(RNGFactory, CreateNormInstance)
{
   unique_ptr<MTRand> rng = RNGFactory::getInstance().createRNG("Norm");
   ASSERT_NE(nullptr, rng);
   ASSERT_NE(nullptr, dynamic_cast<Norm *>(rng.get()));
}

TEST(RNGFactory, CreateNonExistentClassReturnsNullPtr)
{
   unique_ptr<MTRand> rng = RNGFactory::getInstance().createRNG("NonExistent");
   ASSERT_EQ(nullptr, rng);
}