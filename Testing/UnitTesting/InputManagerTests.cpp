/**
 * @file InputManagerTests.cpp
 *
 * @brief This file contains the unit test for the InputManager class
 *
 * @ingroup Testing/UnitTesting
 */

#include "CircularBuffer.h"
#include "InputEvent.h"
#include "InputManager.h"
#include "ParameterManager.h"
#include "gtest/gtest.h"
#include <vector>

using namespace std;
string configFile = "../configfiles/test-spd-911.xml";
string inputs911 = "../Testing/TestData/NG_911_inputs.xml";
string neuroInputs = "../Testing/TestData/neuro_inputs.xml";

class InputManagerFixture : public ::testing::Test {
public:
   InputManager<Call> inputManager;

   void SetUp()
   {
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

TEST_F(InputManagerFixture, queueFront)
{
   Call event = inputManager.queueFront(194);
   EXPECT_EQ(event.vertexId, 194);
   EXPECT_EQ(event.time, 0);
   EXPECT_EQ(event.duration, 0);
   EXPECT_EQ(event.type, "EMS");
   EXPECT_DOUBLE_EQ(event.x, -122.38496236371942);
   EXPECT_DOUBLE_EQ(event.y, 47.570236838209546);
}

TEST_F(InputManagerFixture, queueFrontException)
{
   try {
      // This should throw an out_of_range exception because
      // we don't have input calls for vertexId 35
      Call call = inputManager.queueFront(35);
      FAIL() << "Expected std::out_of_range";
   } catch (std::out_of_range const &err) {
      EXPECT_EQ(err.what(), std::string("map::at"));
   } catch (...) {
      FAIL() << "Expected std::out_of_range";
   }
}

TEST_F(InputManagerFixture, getEpochEvents)
{
   // Get events for vertex 194 between time step 0 (inclusive) and 37 (exclusive)
   CircularBuffer<Call> v194Queue(5);   // There are only 4 inputs per vertex
   v194Queue = inputManager.getEvents(194, 0, 37, v194Queue);
   // There should be 2 events in the list
   ASSERT_EQ(v194Queue.size(), 2);

   // Check that we ge the correct information for the first
   // expected event
   std::optional<Call> call1 = v194Queue.get();
   ASSERT_TRUE(call1);
   EXPECT_EQ(call1->vertexId, 194);
   EXPECT_EQ(call1->time, 0);
   EXPECT_EQ(call1->duration, 0);
   EXPECT_EQ(call1->type, "EMS");
   EXPECT_DOUBLE_EQ(call1->x, -122.38496236371942);
   EXPECT_DOUBLE_EQ(call1->y, 47.570236838209546);

   // Check that we ge the correct information for the second
   // expected event
   std::optional<Call> call2 = v194Queue.get();
   ASSERT_TRUE(call2);
   EXPECT_EQ(call2->vertexId, 194);
   EXPECT_EQ(call2->time, 34);
   EXPECT_EQ(call2->duration, 230);
   EXPECT_EQ(call2->type, "EMS");
   EXPECT_DOUBLE_EQ(call2->x, -122.37482094435583);
   EXPECT_DOUBLE_EQ(call2->y, 47.64839548276973);

   // Get event for vertex 195
   CircularBuffer<Call> v195Queue(5);
   v195Queue = inputManager.getEvents(195, 125, 401, v195Queue);

   ASSERT_EQ(v195Queue.size(), 3);   // should not include 401
   // Check that 3rd event is correct
   // pop and discard the first 2 events
   v195Queue.get();
   v195Queue.get();
   std::optional<Call> call3 = v195Queue.get();
   EXPECT_EQ(call3->vertexId, 195);
   EXPECT_EQ(call3->time, 388);
   EXPECT_EQ(call3->duration, 45);
   EXPECT_EQ(call3->type, "Law");
   EXPECT_DOUBLE_EQ(call3->x, -122.37746466732693);
   EXPECT_DOUBLE_EQ(call3->y, 47.711139673719046);
}

TEST_F(InputManagerFixture, nonEmptyEventQueue)
{
   ASSERT_FALSE(inputManager.queueEmpty(194));
   ASSERT_FALSE(inputManager.queueEmpty(195));
}

TEST_F(InputManagerFixture, emptyQueueAfterGettingEpochEvents)
{
   ASSERT_FALSE(inputManager.queueEmpty(194));
   // pop all 4 events for 194
   CircularBuffer<Call> testQueue(4);
   testQueue = inputManager.getEvents(194, 0, 74, testQueue);
   ASSERT_EQ(testQueue.size(), 4);

   // Queue shold now be empty
   ASSERT_TRUE(inputManager.queueEmpty(194));
}

TEST_F(InputManagerFixture, emptyQueueForMissingVertex)
{
   ASSERT_TRUE(inputManager.queueEmpty(35));
}

TEST_F(InputManagerFixture, queuePop)
{
   ASSERT_FALSE(inputManager.queueEmpty(194));

   while (!inputManager.queueEmpty(194)) {
      inputManager.queuePop(194);
   }

   ASSERT_TRUE(inputManager.queueEmpty(194));
}

TEST_F(InputManagerFixture, clockTickSize)
{
   ASSERT_EQ(inputManager.getClockTickSize(), 1);
}

TEST_F(InputManagerFixture, clockTickUnit)
{
   ASSERT_EQ(inputManager.getClockTickUnit(), "sec");
}

TEST(InputManager, readNeuroInputs)
{
   InputManager<InputEvent> inputManager;
   inputManager.setInputFilePath(neuroInputs);

   inputManager.registerProperty("vertex_id", &InputEvent::vertexId);
   inputManager.registerProperty("time", &InputEvent::time);

   inputManager.readInputs();

   ASSERT_FALSE(inputManager.queueEmpty(1));
   CircularBuffer<InputEvent> testQueue(4);
   inputManager.getEvents(1, 0, 74, testQueue);
   ASSERT_EQ(testQueue.size(), 4);

   // Check all events in the list
   std::optional<InputEvent> iEvent1 = testQueue.get();
   ASSERT_TRUE(iEvent1);
   ASSERT_EQ(iEvent1.value().vertexId, 1);
   ASSERT_EQ(iEvent1.value().time, 0);

   std::optional<InputEvent> iEvent2 = testQueue.get();
   ASSERT_TRUE(iEvent2);
   ASSERT_EQ(iEvent2.value().vertexId, 1);
   ASSERT_EQ(iEvent2.value().time, 34);

   auto iEvent3 = testQueue.get();
   ASSERT_TRUE(iEvent3);
   ASSERT_EQ(iEvent3->vertexId, 1);
   ASSERT_EQ(iEvent3->time, 47);

   auto iEvent4 = testQueue.get();
   ASSERT_TRUE(iEvent4);
   ASSERT_EQ((*iEvent4).vertexId, 1);
   ASSERT_EQ((*iEvent4).time, 73);

   ASSERT_EQ(inputManager.getClockTickSize(), 100);
   ASSERT_EQ(inputManager.getClockTickUnit(), "usec");
}