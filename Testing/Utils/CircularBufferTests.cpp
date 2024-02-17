/**
 * @file CircularBufferTests.cpp
 * 
 * @brief Unit Tests for CircularBuffer class.
 * 
 * @ingroup Testing/Utils
 * 
 */


#include "CircularBuffer.h"
#include "gtest/gtest.h"
#include <string>
#include <vector>

struct TestStruct {
   int testInt;
   double testDouble;
   std::string testString;

   bool operator==(const TestStruct &rhs) const
   {
      return testInt == rhs.testInt && testDouble == testDouble && testString == rhs.testString;
   }
};

// Tests that the Circular Buffer is initialized as expected
TEST(CircularBuffer, Constructor)
{
   CircularBuffer<int> cbInt(11);
   ASSERT_TRUE(cbInt.isEmpty());
   ASSERT_FALSE(cbInt.isFull());
   ASSERT_EQ(11, cbInt.capacity());
   ASSERT_EQ(0, cbInt.size());
}

// Tests that we can resize and empty buffer
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
   ASSERT_EQ(0, cbResizable.size());

   // Resize back to 0
   cbResizable.resize(0);
   ASSERT_EQ(0, cbResizable.capacity());
}

// Tests that put adds the correct number of elements
TEST(CircularBuffer, Put)
{
   CircularBuffer<TestStruct> testBuffer(5);

   // Set the test structs
   std::vector<TestStruct> tsVector(5);
   tsVector[0] = {1, 1.1, "test1"};
   tsVector[1] = {2, 2.2, "test2"};
   tsVector[2] = {3, 3.3, "test3"};
   tsVector[3] = {4, 4.4, "test4"};
   tsVector[4] = {5, 5.5, "test5"};

   // Buffer is empty to start with
   ASSERT_TRUE(testBuffer.isEmpty());
   ASSERT_EQ(0, testBuffer.size());
   ASSERT_FALSE(testBuffer.isFull());
   ASSERT_EQ(5, testBuffer.capacity());

   // Insert struct one by one into the buffer
   int size = 0;
   for (TestStruct ts : tsVector) {
      testBuffer.put(ts);
      size++;   // size should have increased by 1
      ASSERT_EQ(size, testBuffer.size());
      ASSERT_FALSE(testBuffer.isEmpty());
   }

   // Since we added 5 elements, the buffer should be full now
   ASSERT_TRUE(testBuffer.isFull());
}

// Tests that get retrieves the correct value
TEST(CircularBuffer, Get)
{
   int maxSize = 10;

   CircularBuffer<int> testBuffer(maxSize);
   // Buffer is empty to start with
   ASSERT_TRUE(testBuffer.isEmpty());
   ASSERT_EQ(0, testBuffer.size());
   ASSERT_FALSE(testBuffer.isFull());
   ASSERT_EQ(maxSize, testBuffer.capacity());

   // We will insert values 0 through 9
   for (int i = 0; i < maxSize; i++) {
      testBuffer.put(i);
   }

   // The buffer is now full
   ASSERT_EQ(maxSize, testBuffer.size());
   ASSERT_TRUE(testBuffer.isFull());

   // We will get the values one by one and check
   int size = testBuffer.size();
   for (int i = 0; i < maxSize; i++) {
      std::optional<int> val = testBuffer.get();
      ASSERT_TRUE(val);
      ASSERT_EQ(i, *val);
      size--;   // Size should be reduced by 1
      ASSERT_EQ(size, testBuffer.size());
   }

   // The buffer should now be empty
   ASSERT_TRUE(testBuffer.isEmpty());
}

// Tests that get retrieves correct value when the buffer is empty
TEST(CircularBuffer, GetWhenEmpty)
{
   CircularBuffer<TestStruct> testBuffer(5);

   // Given an empty buffer
   ASSERT_TRUE(testBuffer.isEmpty());
   // When we try to get the value at the front of te queue
   std::optional<TestStruct> val = testBuffer.get();
   // Then the object contained int the optional object is invalid
   ASSERT_FALSE(val);
}

// Tests that the buffer is effectively empty after clear() is called
TEST(CircularBuffer, Clear)
{
   int maxSize = 100;
   CircularBuffer<int> testBuffer(maxSize);

   // We will add a few values and then clear the buffer
   int fillSize = 10;
   for (int i = 0; i < fillSize; i++) {
      testBuffer.put(i);
   }

   // Check the new size, empty and full state
   ASSERT_EQ(fillSize, testBuffer.size());
   ASSERT_FALSE(testBuffer.isEmpty());
   ASSERT_FALSE(testBuffer.isFull());
   ASSERT_EQ(maxSize, testBuffer.capacity());

   // Let's clear the buffer
   testBuffer.clear();
   ASSERT_TRUE(testBuffer.isEmpty());
   ASSERT_FALSE(testBuffer.isFull());
   ASSERT_EQ(0, testBuffer.size());
   ASSERT_EQ(maxSize, testBuffer.capacity());

   // Let's try to add a new value now
   testBuffer.put(42);
   ASSERT_FALSE(testBuffer.isEmpty());
}

// Tests that isEmpty return value is correct
TEST(CircularBuffer, IsEmpty)
{
   int maxSize = 99;
   CircularBuffer<int> testBuffer(maxSize);

   // We will fill in the buffer and then get each value until it is empty
   for (int i = 0; i < maxSize; i++) {
      testBuffer.put(i);
   }

   ASSERT_TRUE(testBuffer.isFull());

   // Get all value until buffer is empty
   int size = testBuffer.size();
   ASSERT_EQ(maxSize, size);
   while (!testBuffer.isEmpty()) {
      std::optional<int> val = testBuffer.get();
      ASSERT_TRUE(val);   // val should contain a valid value
   }

   ASSERT_TRUE(testBuffer.isEmpty());
}

// Tests that isFull return value is correct
TEST(CircularBuffer, IsFull)
{
   int maxSize = 1000000;   // one million
   CircularBuffer<double> testBuffer(maxSize);

   int curSize = 0;
   while (!testBuffer.isFull()) {
      testBuffer.put(3.14159);
      curSize++;
   }

   ASSERT_EQ(maxSize, testBuffer.size());
   ASSERT_EQ(curSize, testBuffer.capacity());
   ASSERT_TRUE(testBuffer.isFull());
}

// Tests that capacity() returns the correct value
TEST(CircularBuffer, Capacity)
{
   int maxSize = 1000000;
   CircularBuffer<TestStruct> testBuffer(maxSize);
   ASSERT_EQ(maxSize, testBuffer.capacity());
}

// Tests that size() returns the correct value
TEST(CircularBuffer, Size)
{
   int maxSize = 100;
   CircularBuffer<double> testBuffer(maxSize);

   // We will add a few values
   ASSERT_EQ(0, testBuffer.size());
   for (int i = 0; i < maxSize; i++) {
      testBuffer.put(3.14159);
      ASSERT_EQ(i + 1, testBuffer.size());
   }

   // At this point the buffer is full
   ASSERT_TRUE(testBuffer.isFull());
   ASSERT_EQ(maxSize, testBuffer.size());

   // If we remove 2 item and then add 2 more,
   // the last item will fall at the beginning of
   // the buffer. The buffer should still be
   // full and size should equal maxSize.
   testBuffer.get();
   testBuffer.get();
   ASSERT_EQ(maxSize - 2, testBuffer.size());
   testBuffer.put(6.28);
   testBuffer.put(6.28);
   ASSERT_TRUE(testBuffer.isFull());
   ASSERT_EQ(maxSize, testBuffer.size());
}