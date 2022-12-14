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

    // this should throw an exception because vertexId doesn't have events
    EXPECT_ANY_THROW(inputManager.vertexQueueFront(35));
}

// TEST_F(InputManagerFixture, registerProperty) {
//     inputManager.registerProperty("vertex_id", InputManager::PropertyType::INTEGER, &Event::vertexId);
    
//     // int Event::*ptiptr = boost::get<int Event::*>(inputManager.eventMemberPtrMap_["vertex_id"].second);
//     // Event newEvent;
//     // newEvent.*ptiptr = 10;
//     ASSERT_EQ(newEvent.vertexId, 10);
// }