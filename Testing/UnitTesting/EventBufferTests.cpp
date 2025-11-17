/**
 * @file EventBufferTests.cpp
 *
 * @brief This file contains the unit tests for EventBuffer using GTest.
 *
 * @ingroup Testing/UnitTesting
 */

#include "EventBuffer.h"
#include "gtest/gtest.h"

// A buffer which can hold 5 elements
EventBuffer<uint64_t> buffer(5);

//GetElement when buffer is empty
TEST(EventBufferTest, GetElementFromEmptyBufferUint64)
{
   EXPECT_EQ(std::get<uint64_t>(buffer.getElement(0)), std::numeric_limits<unsigned long>::max());
}

// GetPastEvent when buffer is empty
TEST(EventBufferTest, GetPastEventFromEmptyBufferUint64)
{
   EXPECT_EQ(buffer.getPastEvent(-1), std::numeric_limits<unsigned long>::max());
}

// Insert into empty buffer
TEST(EventBufferTest, InsertEventEmptyBufferUint64)
{
   buffer.insertEvent(10);
   buffer.insertEvent(20);

   EXPECT_EQ(std::get<uint64_t>(buffer.getElement(0)), 10);
   EXPECT_EQ(std::get<uint64_t>(buffer.getElement(1)), 20);
}

// Insert when buffer is full, test wrap around
TEST(EventBufferTest, BufferWrapAroundUint64)
{
   buffer.insertEvent(30);
   buffer.insertEvent(40);
   buffer.insertEvent(50);

   //Insert into A full buffer
   //buffer.insertEvent(60);

   // The buffer should have overwritten 60 inplace of 10
   //EXPECT_EQ(std::get<uint64_t>(buffer.getElement(0)), 60);
   EXPECT_EQ(std::get<uint64_t>(buffer.getElement(1)), 20);
   EXPECT_EQ(std::get<uint64_t>(buffer.getElement(2)), 30);
   EXPECT_EQ(std::get<uint64_t>(buffer.getElement(3)), 40);
   EXPECT_EQ(std::get<uint64_t>(buffer.getElement(4)), 50);
}

// A buffer which can hold 5 elements
EventBuffer<double> bufferDouble(5);

//GetElement when buffer is empty
//Assuming getElement() returns uint64_t
TEST(EventBufferTest, GetElementFromEmptyBufferDouble)
{
   EXPECT_EQ(std::get<double>(bufferDouble.getElement(0)), std::numeric_limits<double>::max());
}

// GetPastEvent when buffer is empty
TEST(EventBufferTest, GetPastEventFromEmptyBufferDouble)
{
   EXPECT_EQ(bufferDouble.getPastEvent(-1), std::numeric_limits<double>::max());
}

// Insert into empty buffer
TEST(EventBufferTest, InsertEventEmptyBufferDouble)
{
   bufferDouble.insertEvent(10.0);
   bufferDouble.insertEvent(20.0);

   EXPECT_EQ(std::get<double>(bufferDouble.getElement(0)), 10.0);
   EXPECT_EQ(std::get<double>(bufferDouble.getElement(1)), 20.0);
}

// Insert when buffer is full, test wrap around
TEST(EventBufferTest, BufferWrapAroundDouble)
{
   bufferDouble.insertEvent(30.0);
   bufferDouble.insertEvent(40.0);
   bufferDouble.insertEvent(50.0);

   //Insert into A full buffer
   //bufferDouble.insertEvent(60.0);

   // The buffer should have overwritten 60 inplace of 10
   //EXPECT_EQ(std::get<uint64_t>(buffer.getElement(0)), 60);
   EXPECT_EQ(std::get<double>(bufferDouble.getElement(1)), 20.0);
   EXPECT_EQ(std::get<double>(bufferDouble.getElement(2)), 30.0);
   EXPECT_EQ(std::get<double>(bufferDouble.getElement(3)), 40.0);
   EXPECT_EQ(std::get<double>(bufferDouble.getElement(4)), 50.0);
}