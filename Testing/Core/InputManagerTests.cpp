/**
 * @file InputManagerTests.cpp
 *
 * @brief This file contains the unit test for the InputManager class
 *
 * @ingroup Testing/Core
 */

#include <vector>
#include "ParameterManager.h"
#include "InputManager.h"
#include "Event.h"
#include "gtest/gtest.h"

using namespace std;
string configFile = "../configfiles/test-spd-911.xml";
string inputs911 = "../Testing/TestData/NG_911_inputs.xml";
string neuroInputs = "../Testing/TestData/neuro_inputs.xml";

class InputManagerFixture: public ::testing::Test {
public:
    InputManager<Call> inputManager;

    void SetUp() {
        inputManager.setInputFilePath(inputs911);
        inputManager.registerProperty("vertex_id", &Call::vertexId);
        inputManager.registerProperty("time", &Call::time);
        inputManager.registerProperty("duration", &Call::duration);
        inputManager.registerProperty("x", &Call::x);
        inputManager.registerProperty("y", &Call::y);
        inputManager.registerProperty("type", &Call::type);

        inputManager.readInputs();
    }
};

TEST_F(InputManagerFixture, queueFront) {
    Call event = inputManager.queueFront(194);
    EXPECT_EQ(event.vertexId, 194);
    EXPECT_EQ(event.time, 0);
    EXPECT_EQ(event.duration, 0);
    EXPECT_EQ(event.type, "EMS");
    EXPECT_DOUBLE_EQ(event.x, -122.38496236371942);
    EXPECT_DOUBLE_EQ(event.y, 47.570236838209546);
}

TEST_F(InputManagerFixture, queueFrontException) {
    try {
        // This should throw an out_of_range exception because
        // we don't have input calls for vertexId 35
        Call call = inputManager.queueFront(35);
        FAIL() << "Expected std::out_of_range";
    } catch (std::out_of_range const &err) {
        EXPECT_EQ(err.what(),std::string("map::at"));
    } catch (...) {
        FAIL() << "Expected std::out_of_range";
    }
}

TEST_F(InputManagerFixture, getEpochEvents) {
    // Get events for vertex 194 between time step 0 (inclusive) and 37 (exclusive)
    vector<Call> epochEventList = inputManager.getEvents(194, 0, 37);
    // There should be 2 events in the list
    ASSERT_EQ(epochEventList.size(), 2);

    // Check that we ge the correct information for the first
    // expected event
    EXPECT_EQ(epochEventList[0].vertexId, 194);
    EXPECT_EQ(epochEventList[0].time, 0);
    EXPECT_EQ(epochEventList[0].duration, 0);
    EXPECT_EQ(epochEventList[0].type, "EMS");
    EXPECT_DOUBLE_EQ(epochEventList[0].x, -122.38496236371942);
    EXPECT_DOUBLE_EQ(epochEventList[0].y, 47.570236838209546);

    // Check that we ge the correct information for the second
    // expected event
    EXPECT_EQ(epochEventList[1].vertexId, 194);
    EXPECT_EQ(epochEventList[1].time, 34);
    EXPECT_EQ(epochEventList[1].duration, 230);
    EXPECT_EQ(epochEventList[1].type, "EMS");
    EXPECT_DOUBLE_EQ(epochEventList[1].x, -122.37482094435583);
    EXPECT_DOUBLE_EQ(epochEventList[1].y, 47.64839548276973);

    // Get event for vertex 195
    auto v195Events = inputManager.getEvents(195, 125, 401);

    ASSERT_EQ(v195Events.size(), 3);    // should not include 401
    // Check that 3rd event is correct
    EXPECT_EQ(v195Events[2].vertexId, 195);
    EXPECT_EQ(v195Events[2].time, 388);
    EXPECT_EQ(v195Events[2].duration, 45);
    EXPECT_EQ(v195Events[2].type, "Law");
    EXPECT_DOUBLE_EQ(v195Events[2].x, -122.37746466732693);
    EXPECT_DOUBLE_EQ(v195Events[2].y, 47.711139673719046);
}

TEST_F(InputManagerFixture, nonEmptyEventQueue) {
    ASSERT_FALSE(inputManager.queueEmpty(194));
    ASSERT_FALSE(inputManager.queueEmpty(195));
}

TEST_F(InputManagerFixture, emptyQueueAfterGettingEpochEvents) {
    ASSERT_FALSE(inputManager.queueEmpty(194));
    // pop all 4 events for 194
    auto eventList = inputManager.getEvents(194, 0, 74);
    ASSERT_EQ(eventList.size(), 4);

    // Queue shold now be empty
    ASSERT_TRUE(inputManager.queueEmpty(194));
}

TEST_F(InputManagerFixture, emptyQueueForMissingVertex) {
    ASSERT_TRUE(inputManager.queueEmpty(35));
}

TEST_F(InputManagerFixture, queuePop) {
    ASSERT_FALSE(inputManager.queueEmpty(194));

    while (!inputManager.queueEmpty(194)) {
        inputManager.queuePop(194);
    }

    ASSERT_TRUE(inputManager.queueEmpty(194));
}

TEST(InputManagerFixure, invalidInputFilePath) {
    
}


TEST(InputManager, readNeuroInputs) {
    InputManager<Event> inputManager;
    inputManager.setInputFilePath(neuroInputs);

    inputManager.registerProperty("vertex_id", &Event::vertexId);
    inputManager.registerProperty("time", &Event::time);

    inputManager.readInputs();

    ASSERT_FALSE(inputManager.queueEmpty(1));
    auto eventList = inputManager.getEvents(1, 0, 74);
    ASSERT_EQ(eventList.size(), 4);

    // Check all events in the list
    ASSERT_EQ(eventList[0].vertexId, 1);
    ASSERT_EQ(eventList[0].time, 0);

    ASSERT_EQ(eventList[1].vertexId, 1);
    ASSERT_EQ(eventList[1].time, 34);

    ASSERT_EQ(eventList[2].vertexId, 1);
    ASSERT_EQ(eventList[2].time, 47);

    ASSERT_EQ(eventList[3].vertexId, 1);
    ASSERT_EQ(eventList[3].time, 73);
}