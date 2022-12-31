
#pragma once

#include "ParameterManager.h"
#include <boost/variant.hpp>
#include <log4cplus/loggingmacros.h>
#include <map>
#include <queue>

#include <boost/foreach.hpp>
#include <boost/property_tree/exceptions.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <iostream>

using namespace std;
using boost::property_tree::ptree;

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

template <typename T>
class InputManager {
public:
   // Set boost::variant to store the following types. We will use variant.which() to
   // determine the currently stored value at runtime. Which returns an index value
   // based on the order in which the type appears in the list, in this case:
   // int Event::* = 0, uint64_t Event::* = 1 ... string Event::* = 5
   // For convenience an enum type is defined below but the types must be in the same
   // order as declared in the boost::variant typedef.
   using EventMemberPtr = boost::variant<int T::*, uint64_t T::*, long T::*, float T::*,
                          double T::*, string T::*>;
   enum PropertyType { INTEGER, UINT64, LONG, FLOAT, DOUBLE, STRING };

   // Some aliases for better readability
   using VertexId_t = int;
   using EventMap_t = map<VertexId_t, queue<T>>;

   /// @brief  Constructor
   InputManager() {
      // Get a copy of the file logger to use with log4cplus macros
      fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));
      LOG4CPLUS_DEBUG(fileLogger_, "Initializing InputManager");
   }

   void setInputFilePath(const string &inputFilePath) {
      if (inputFilePath.empty()) {
         // TODO: terminate if an a empty string is given as inputFilePath
         LOG4CPLUS_FATAL(fileLogger_, "inputFilePath must not be an empty string");
         exit(EXIT_FAILURE);
      }

      inputFilePath_ = inputFilePath;
   }

   /// @brief  Read a list of events from an input file and load them into
   ///         a map, organized per vertex ID
   /// @return True if the file was successfully read, false otherwise
   bool readInputs() {
      ptree pt;
      ifstream inputFile;
      // string inputFilePath;

      if (inputFilePath_.empty()) {
         // If no inputFilePath_ has bee assigned get it from the ParameterManager
         string xpath = "//inputFile/text()";
         if (!ParameterManager::getInstance().getStringByXpath(xpath, inputFilePath_)) {
            cerr << "InputManager: Count not find XML Path: " << xpath << ".\n";
            return false;
         }
      }

      assert(!inputFilePath_.empty());

      inputFile.open(inputFilePath_.c_str());
      if (!inputFile.is_open()) {
         cerr << "InputManager: Failed to open file: " << inputFilePath_ << ".\n";
         return false;
      }

      boost::property_tree::xml_parser::read_xml(inputFile, pt);
      BOOST_FOREACH (ptree::value_type const &v, pt.get_child("simulator_inputs").get_child("data")) {
         if (v.first == "vertex") {
            int vertex_id = v.second.get_child("<xmlattr>").get<int>("id");
            // loop over list of events that belong to this vertex
            BOOST_FOREACH (ptree::value_type const &evnt, v.second) {
               if (evnt.first == "event") {
                  try {
                     const ptree &attributes = evnt.second.get_child("<xmlattr>");
                     T event;
                     // Loop over the set of attributes and add them to the event object.
                     // We only read registered properties, all others are ignored.
                     for (auto propPair : registeredPropMap_) {
                        getProperty(event, propPair.first, propPair.second, attributes);
                     }
                     // Add the event object to the event map. Map's operator[] creates
                     // and empty queue if it doesn't yet contain this vertex_id.
                     eventsMap_[vertex_id].push(event);
                  } catch (boost::property_tree::ptree_bad_data e) {
                     LOG4CPLUS_FATAL(fileLogger_, "InputManager failed to read event node: " << e.what());
                     // TODO: perhaps we need to exit with a fatal log message since there is
                     // missing information
                     return false;
                  } catch (boost::bad_get e) {
                     LOG4CPLUS_FATAL(fileLogger_, "Failed to read event property: " << e.what());
                     return false;
                  }
               }
            }
         }
      }
      LOG4CPLUS_DEBUG(fileLogger_, "Input file loaded successfully");
      return true;
   }

   /// @brief  Retrieves a list of events that occur between firstStep (inclusive) and
   ///         lastStep (exclusive) in the given vertexId.
   /// @param vertexId     The ID of the vertex where the events occur
   /// @param firstStep    The first time step (inclusive) for the occurrence of the events
   /// @param lastStep     The last time step (exclusive) for the occurrence of the events
   /// @return The list of events between firstStep and lastStep for the fiven vertexId
   vector<T> getEvents(const VertexId_t &vertexId, uint64_t firstStep, uint64_t lastStep) {
      vector<Event> result = vector<Event>();   // Will hold the list of events
      queue<Event> &eventQueue = eventsMap_[vertexId];   // Get a reference to the event queue

      while (!eventQueue.empty() && eventQueue.front().time < lastStep) {
         // We shouldn't have previous epoch events in the queue
         assert(eventQueue.front().time >= firstStep);
         result.push_back(eventQueue.front());
         eventQueue.pop();
      }

      return result;
   }

   /// @brief  Peeks into the event at the front of the vertex queue
   /// @param vertexId  The ID of the vertex
   /// @return    The event at the front of the given vertex queue
   Event vertexQueueFront(const VertexId_t &vertexId) {
      // TODO: this throws an out_of_range exception if vertexId is not found
      return eventsMap_.at(vertexId).front();
   }

   /// @brief  Removes the event at the back of the vertex queue
   /// @param vertexId  The ID of the vertex
   void vertexQueuePop(const VertexId_t &vertexId) {
      // TODO: handle when vertexID is not found in the map
      eventsMap_.at(vertexId).pop();
   }

   /// @brief  Registers a property with the given name and a pointer to a member of the
   ///         Event class. The pointer to member is stored in a boost::variant type that
   ///         is later used to assign the input event data to the correct member variable
   /// @param propName  The name of the property as defined in the input file
   /// @param property  The pointer to member variable where the property should be stored
   /// @return    True if the property was successfully registered, false otherwise
   bool registerProperty(const string &propName, EventMemberPtr property) {
      // Check that we the propName is not an empty string
      if (propName.empty()) {
         LOG4CPLUS_ERROR(fileLogger_, "Property name should not be an empty string");
         return false;
      }

      // The compiler will check that only a type that is part of our variant can be
      // registered.
      registeredPropMap_[propName] = property;
      return true;
   }

private:
   // Map that stores a queue of events per vertex
   map<VertexId_t, queue<T>> eventsMap_;

   // The path to the input file
   string inputFilePath_;

   // Structures for Dynamically registering the Event properties
   // We could specify the types in the input file but we don't have to
   // if we make sure that the string can be casted to the registered type.
   // map<propName, ptrToMember>
   map<string, EventMemberPtr> registeredPropMap_;

   log4cplus::Logger fileLogger_;

   bool getProperty(T &event, string propName, EventMemberPtr &eventMbrPtr,
                    const boost::property_tree::ptree &pTree) {
      switch (eventMbrPtr.which()) {
         // variant.which() returns a value between 0 and number of types - 1
         case PropertyType::INTEGER: {
            int T::*propPtr = get<int T::*>(eventMbrPtr);
            event.*propPtr = pTree.get<int>(propName);
         } break;
         case PropertyType::LONG: {
            long T::*propPtr = get<long T::*>(eventMbrPtr);
            event.*propPtr = pTree.get<long>(propName);
         } break;
         case PropertyType::UINT64: {
            uint64_t T::*propPtr = get<uint64_t T::*>(eventMbrPtr);
            event.*propPtr = pTree.get<uint64_t>(propName);
         } break;
         case PropertyType::FLOAT: {
            float T::*propPtr = get<float T::*>(eventMbrPtr);
            event.*propPtr = pTree.get<float>(propName);
         } break;
         case PropertyType::DOUBLE: {
            double T::*propPtr = get<double T::*>(eventMbrPtr);
            event.*propPtr = pTree.get<double>(propName);
         } break;
         case PropertyType::STRING: {
            string T::*propPtr = get<string T::*>(eventMbrPtr);
            event.*propPtr = pTree.get<string>(propName);
         } break;
         default:
            LOG4CPLUS_DEBUG(fileLogger_, "Property not supported: " << propName);
            return false;
      }
      return true;
   }
};
