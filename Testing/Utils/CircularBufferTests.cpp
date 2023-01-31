

#include <string.h>

#include "CircularBuffer.h"
#include "gtest/gtest.h"

struct TestSruct {
   int testInt;
   double testDouble;
   std::string testString;
};

// Tests that the Circular Buffer is initialized as expected
TEST(CircularBuffer, Constructor)
{
   CircularBuffer<int> cbInt(11);
   ASSERT_TRUE(cbInt.isEmpty());
   ASSERT_FALSE(cbInt.isFull());
   ASSERT_EQ(11, cbInt.capacity());
}

TEST(CircularBuffer, ConstructAndResize)
{
   CircularBuffer<int> cbResizable;   // Buffer of size 0
   ASSERT_EQ(0, cbResizable.capacity());
   ASSERT_TRUE(cbResizable.isEmpty());

   // Resize to 15
   cbResizable.resize(15);
   ASSERT_TRUE(cbResizable.isEmpty());
   ASSERT_FALSE(cbResizable.isFull());
   ASSERT_EQ(15, cbResizable.capacity());
}

// Tests that put adds the correct number of elements
TEST(CircularBuffer, Put)
{

}

// Tests that get retrieves the correct value
TEST(CircularBuffer, Get)
{

}

// Tests that get retrieves correct value when the buffer is empty
TEST(CircularBuffer, GetWhenEmpty)
{

}

// Tests that we can resize an empty buffer
TEST(CircularBuffer, Resize)
{

}

// Tests that the buffer is effectively empty after clear() is called
TEST(CircularBuffer, Clear)
{

}

// Tests that isEmpty return value is correct
TEST(CircularBuffer, IsEmpty)
{

}

// Tests that isFull return value is correct
TEST(CircularBuffer, IsFull)
{

}

// Tests that capacity() returns the correct value
TEST(CircularBuffer, Capacity)
{

}

// Tests that size() returns the correct value
TEST(CircularBuffer, Size)
{

}