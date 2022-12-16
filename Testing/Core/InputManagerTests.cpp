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
#include "gtest/gtest.h"

using namespace std;
string configFile = "../configfiles/test-spd-911.xml";

class InputManagerFixture: public ::testing::Test {
public:
    InputManager inputManager;

    void SetUp() {
        bool areParamsLoaded = ParameterManager::getInstance().loadParameterFile(configFile);
        ASSERT_TRUE(areParamsLoaded);
        inputManager.registerProperty("vertex_id", &Event::vertexId);
        inputManager.registerProperty("time", &Event::time);
        inputManager.registerProperty("duration", &Event::duration);
        inputManager.registerProperty("x", &Event::x);
        inputManager.registerProperty("y", &Event::y);
        inputManager.registerProperty("type", &Event::type);

        bool areInputsRead = inputManager.readInputs();
        ASSERT_TRUE(areInputsRead);
    }
};

// TEST_F(InputManagerFixture, ReadInputFile) {
//     vector<Event> events = inputManager.getEvents();
//     size_t numEvents = events.size();
//     ASSERT_EQ(numEvents, 41217);
// }

TEST_F(InputManagerFixture, vertexQueueFront) {
    Event event = inputManager.vertexQueueFront(194);
    EXPECT_EQ(event.vertexId, 194);
    EXPECT_EQ(event.time, 0);
    EXPECT_EQ(event.duration, 0);
    EXPECT_EQ(event.type, "EMS");
    EXPECT_DOUBLE_EQ(event.x, -122.38496236371942);
    EXPECT_DOUBLE_EQ(event.y, 47.570236838209546);

    // this throws an exception because vertexId doesn't have events
    // TODO: return an empty vector instead
    EXPECT_ANY_THROW(inputManager.vertexQueueFront(35));
}

TEST_F(InputManagerFixture, getEpochEvents) {
    // Get events between time step 0 (inclusive) and 37 (exclusive)
    vector<Event> epochEventList = inputManager.getEvents(194, 0, 37);
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

}
