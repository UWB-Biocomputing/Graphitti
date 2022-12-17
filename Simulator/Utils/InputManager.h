
#pragma once

#include "ParameterManager.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/variant.hpp>
#include <log4cplus/loggingmacros.h>
#include <map>
#include <queue>

using namespace std;

struct Event {
   // The vertexId where the input event happen
   int vertexId;
   // The start of the event since the beggining of
   // the simulation in timesteps matches g_simulationStep type
   uint64_t time;
   // The duration of the event in timesteps
   int duration;
   // Event location
   double x;
   double y;
   string type;
};

class InputManager {
public:
   // Set boost::variant to store the following types. We will use variant.which() to
   // determine the currently stored value at runtime. Which returns an index value
   // based on the order in which the type appears in the list, in this case:
   // int Event::* = 0, uint64_t Event::* = 1 ... string Event::* = 5
   // For convenience an enum type is defined below but the types must be in the same
   // order as declared in the boost::variant typedef.
   using EventMemberPtr = boost::variant<int Event::*, uint64_t Event::*, long Event::*, float Event::*,
                          double Event::*, string Event::*>;
   enum PropertyType { INTEGER, UINT64, LONG, FLOAT, DOUBLE, STRING };

   // Some aliases for better readability
   using VertexId_t = int;
   using EventMap_t = map<VertexId_t, queue<Event>>;

   /// @brief  Constructor
   InputManager();

   void setInputFilePath(const string &inputFilePath);

   /// @brief  Read a list of events from an input file and load them into
   ///         a map, organized per vertex ID
   /// @return True if the file was successfully read, false otherwise
   bool readInputs();

   /// @brief  Retrieves a list of events that occur between firstStep (inclusive) and
   ///         lastStep (exclusive) in the given vertexId.
   /// @param vertexId     The ID of the vertex where the events occur
   /// @param firstStep    The first time step (inclusive) for the occurrence of the events
   /// @param lastStep     The last time step (exclusive) for the occurrence of the events
   /// @return The list of events between firstStep and lastStep for the fiven vertexId
   vector<Event> getEvents(const VertexId_t &vertexId, uint64_t firstStep, uint64_t lastStep);

   /// @brief  Peeks into the event at the front of the vertex queue
   /// @param vertexId  The ID of the vertex
   /// @return    The event at the front of the given vertex queue
   Event vertexQueueFront(const VertexId_t &vertexId);

   /// @brief  Removes the event at the back of the vertex queue
   /// @param vertexId  The ID of the vertex
   void vertexQueuePop(const VertexId_t &vertexId);

   /// @brief  Registers a property with the given name and a pointer to a member of the
   ///         Event class. The pointer to member is stored in a boost::variant type that
   ///         is later used to assign the input event data to the correct member variable
   /// @param propName  The name of the property as defined in the input file
   /// @param property  The pointer to member variable where the property should be stored
   /// @return    True if the property was successfully registered, false otherwise
   bool registerProperty(const string &propName, EventMemberPtr property);

private:
   // Map that stores a queue of events per vertex
   map<VertexId_t, queue<Event>> eventsMap_;

   // The path to the input file
   string inputFilePath_;

   // Structures for Dynamically registering the Event properties
   // We could specify the types in the input file but we don't have to
   // if we make sure that the string can be casted to the registered type.
   // map<propName, ptrToMember>
   map<string, EventMemberPtr> registeredPropMap_;

   log4cplus::Logger fileLogger_;

   bool getProperty(Event &event, string propName, EventMemberPtr &eventMbrPtr,
                    const boost::property_tree::ptree &pTree);
};