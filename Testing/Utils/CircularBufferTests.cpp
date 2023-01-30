

#include "CircularBuffer.h"
#include "gtest/gtest.h"

// Tests that the Circular Buffer is initialized as expected
TEST(CircularBuffer, Constructor)
{
   CircularBuffer<int> circle;
   ASSERT_FALSE(circle.isFull());
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