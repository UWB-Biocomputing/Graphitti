
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
   // The duration of the even in timesteps
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
   typedef boost::variant<int Event::*, uint64_t Event::*, long Event::*, float Event::*,
                          double Event::*, string Event::*>
      EventMemberPtr;
   enum PropertyType { INTEGER, UINT64, LONG, FLOAT, DOUBLE, STRING };

   // Some aliases for better readability
   using VertexId_t = int;
   using EventMap_t = map<VertexId_t, queue<Event>>;

   InputManager();

   bool readInputs();

   // vector<Event>& getEvents();

   Event vertexQueueFront(const VertexId_t &vertexId);

   const Event &vertexQueueFront(const VertexId_t &vertexId) const;

   void vertexQueuePop(const VertexId_t &vertexId);

   bool registerProperty(const string &propName, EventMemberPtr property);

   

private:
   // vector<Event> events_;
   EventMap_t eventsMap_;

   // Structures for Dynamically registering the Event properties
   // We could specify the types in the input file but we don't have to
   // if we make sure that the string can be casted to the registered type.
   // map<propName, ptrToMember>
   map<string, EventMemberPtr> registeredPropMap_;

   log4cplus::Logger fileLogger_;

   bool getProperty(Event &event, string propName, EventMemberPtr &eventMbrPtr,
                    const boost::property_tree::ptree &pTree);
};