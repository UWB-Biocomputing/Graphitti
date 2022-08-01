/**
 * @file VerticesFactoryTests.cpp
 * 
 * @brief This file contains unit tests for the VerticesFactory using GTest.
 * 
 * @ingroup Testing/Core
 * 
 * We test that the VerticesFactory returns an instance of the correct class
 * we are requesting.
 */

#include "VerticesFactory.h"
#include "All911Vertices.h"
#include "AllIZHNeurons.h"
#include "AllLIFNeurons.h"
#include "gtest/gtest.h"

TEST(VerticesFactory, GetInstanceReturnsInstance)
{
    VerticesFactory *verticesFactory = &VerticesFactory::getInstance();
    ASSERT_NE(nullptr, verticesFactory);
}

TEST(VerticesFactory, CreateAllLIFNeuronsInstance)
{
    shared_ptr<AllVertices> vertices = 
            VerticesFactory::getInstance().createVertices("AllLIFNeurons");
    ASSERT_NE(nullptr, vertices);
    ASSERT_NE(nullptr, dynamic_cast<AllLIFNeurons *>(vertices.get()));
}

TEST(VerticesFactory, CreateAllIZNeuronsInstance)
{
    shared_ptr<AllVertices> vertices = 
            VerticesFactory::getInstance().createVertices("AllIZHNeurons");
    ASSERT_NE(nullptr, vertices);
    ASSERT_NE(nullptr, dynamic_cast<AllIZHNeurons *>(vertices.get()));
}

TEST(VerticesFactory, CreateAll911VerticesInstance)
{
    shared_ptr<AllVertices> vertices = 
            VerticesFactory::getInstance().createVertices("All911Vertices");
    ASSERT_NE(nullptr, vertices);
    ASSERT_NE(nullptr, dynamic_cast<All911Vertices *>(vertices.get()));
}

TEST(VerticesFactory, CreateNonExistentClassReturnsNullPtr)
{
    shared_ptr<AllVertices> vertices = 
            VerticesFactory::getInstance().createVertices("NonExistent");
    ASSERT_TRUE(vertices == nullptr);
}