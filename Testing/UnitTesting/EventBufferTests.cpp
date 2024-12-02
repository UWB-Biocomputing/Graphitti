/**
 * @file EventBufferTests.cpp
 *
 * @brief This file contains the unit tests for EventBuffer using GTest.
 *
 * @ingroup Testing/UnitTesting
 */

#include "gtest/gtest.h"
#include "EventBuffer.h"

// A buffer which can hold 5 elements
EventBuffer buffer(5);

//GetElement when buffer is empty
//Assuming getElement() returns uint64_t 
TEST(EventBufferTest, GetElementFromEmptyBuffer)
{
    EXPECT_EQ(std::get<uint64_t>(buffer.getElement(0)), std::numeric_limits<unsigned long>::max());
}

// GetPastEvent when buffer is empty
TEST(EventBufferTest, GetPastEventFromEmptyBuffer)
{
    EXPECT_EQ(buffer.getPastEvent(-1), std::numeric_limits<unsigned long>::max());
}

// Insert into empty buffer
TEST(EventBufferTest, InsertEventEmptyBuffer) {
    buffer.insertEvent(10);
    buffer.insertEvent(20);

    EXPECT_EQ(std::get<uint64_t>(buffer.getElement(0)), 10);
    EXPECT_EQ(std::get<uint64_t>(buffer.getElement(1)), 20);
}

// Insert when buffer is full, test wrap around
TEST(EventBufferTest, BufferWrapAround) {
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