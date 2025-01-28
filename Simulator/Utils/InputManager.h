/**
 * @file InputManager.h
 * @author Jardi A. M. Jordan (jardiamj@gmail.com)
 * @date 01-03-2023
 * Supervised by Dr. Michael Stiber, UW Bothell CSSE Division
 * @ingroup Simulator/Utils
 * 
 * @brief Template class for reading input events from XML formatted files.
 * 
 * This class is designed to read input events from input files that contains
 * a list of events in an XML format, organized by the vertex where the events
 * ocurr. The following is an example of an input file for the NG-911 model:
 * 
 * <?xml version='1.0' encoding='UTF-8'?>
 * <simulator_inputs>
 *   <data description="SPD_calls_sept2020" clock_tick_size="1" clock_tick_unit="sec">
 *     <vertex id="194" name="SEATTLE PD Caller region">
 *       <event time="0" duration="0" x="-122.38496236371942" y="47.570236838209546" type="EMS" vertex_id="194"/>
 *       <event time="34" duration="230" x="-122.37482094435583" y="47.64839548276973" type="EMS" vertex_id="194"/>
 *       <event time="37" duration="169" x="-122.4036487601129" y="47.55833788618255" type="Fire" vertex_id="194"/>
 *       <event time="42" duration="327" x="-122.38534886929502" y="47.515324716436346" type="Fire" vertex_id="194"/>
 *     </vertex>
 *   </data>
 * </simulator_inputs>
 * 
 * Similarly to GraphManger, the InputManager is designed so that consumer code
 * can register event attributes to member of a Struct. The inputs will be loaded
 * into a map where the key is the vertex associated with the event and the data
 * a queue of events. See the InputManager unit tests for examples.
 * 
 * 
 *
 */

#pragma once

#include "CircularBuffer.h"
#include "ParameterManager.h"
#include <boost/foreach.hpp>
#include <boost/property_tree/exceptions.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/variant.hpp>
#include <iostream>
#include <log4cplus/loggingmacros.h>
#include <map>
#include <queue>

using namespace std;
using boost::property_tree::ptree;

template <typename T> class InputManager {
public:
   // Set boost::variant to store the following types. We will use variant.which() to
   // determine the currently stored value at runtime. Which returns an index value
   // based on the order in which the type appears in the list, in this case:
   // int Event::* = 0, uint64_t Event::* = 1 ... string Event::* = 5
   // For convenience an enum type is defined below but the types must be in the same
   // order as declared in the boost::variant below.
   using EventMemberPtr
      = boost::variant<int T::*, uint64_t T::*, long T::*, float T::*, double T::*, string T::*>;
   enum class PropertyType { INTEGER, UINT64, LONG, FLOAT, DOUBLE, STRING };

   // Some aliases for better readability
   using VertexId_t = int;
   using EventMap_t = map<VertexId_t, queue<T>>;

   /// @brief  Constructor
   InputManager()
   {
      // Get a copy of the file logger to use with log4cplus macros
      fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));
      consoleLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("console"));
      LOG4CPLUS_DEBUG(fileLogger_, "Initializing InputManager");
   }

   /// @brief Sets the path to the input file
   /// @param inputFilePath   The path to the input file
   void setInputFilePath(const string &inputFilePath)
   {
      if (inputFilePath.empty()) {
         LOG4CPLUS_FATAL(consoleLogger_, "inputFilePath must not be an empty string");
         exit(EXIT_FAILURE);
      }

      inputFilePath_ = inputFilePath;
   }

   /// @brief  Read a list of events from an input file and load them into
   ///         a map, organized per vertex ID
   void readInputs()
   {
      ptree pt;
      ifstream inputFile;

      if (inputFilePath_.empty()) {
         // If no inputFilePath_ has been assigned get it from the ParameterManager
         string xpath = "//inputFile/text()";
         if (!ParameterManager::getInstance().getStringByXpath(xpath, inputFilePath_)) {
            LOG4CPLUS_FATAL(consoleLogger_, "Could not find XML Path: " << xpath);
            exit(EXIT_FAILURE);
         }
      }

      assert(!inputFilePath_.empty());

      inputFile.open(inputFilePath_.c_str());
      if (!inputFile.is_open()) {
         LOG4CPLUS_FATAL(consoleLogger_, "Faile to open file: " << inputFilePath_);
         exit(EXIT_FAILURE);
      }

      // Load all events from the Input File
      boost::property_tree::xml_parser::read_xml(inputFile, pt);

      // Get clock tick size
      const ptree &dataNode = pt.get_child("simulator_inputs").get_child("data");
      const ptree &dataAttr = dataNode.get_child("<xmlattr>");
      clockTickSize_ = dataAttr.get<int>("clock_tick_size");
      clockTickUnit_ = dataAttr.get<string>("clock_tick_unit");

      BOOST_FOREACH (ptree::value_type const &v, dataNode) {
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
                     LOG4CPLUS_FATAL(consoleLogger_,
                                     "InputManager failed to read event node: " << e.what());
                     exit(EXIT_FAILURE);
                  } catch (boost::bad_get e) {
                     LOG4CPLUS_FATAL(consoleLogger_, "Failed to read event property: " << e.what());
                     exit(EXIT_FAILURE);
                  }
               }
            }
         }
      }
      LOG4CPLUS_DEBUG(fileLogger_, "Input file loaded successfully");
   }

   /// @brief  Inserts into a the CircularBuffer output parameter the list of events that
   ///         occur between firstStep (inclusive) and lastStep (exclusive) in the given vertexId.
   /// @param vertexId     The ID of the vertex where the events occur
   /// @param firstStep    The first time step (inclusive) for the occurrence of the events
   /// @param lastStep     The last time step (exclusive) for the occurrence of the events
   /// @param buffer       The CircularBuffer where input events will be inserted
   /// @return A reference to the CircularBuffer containing the list of events between
   ///         firstStep and lastStep for the given vertexId
   CircularBuffer<T> &getEvents(const VertexId_t &vertexId, uint64_t firstStep, uint64_t lastStep,
                                CircularBuffer<T> &buffer)
   {
      queue<T> &eventQueue = eventsMap_[vertexId];   // Get a reference to the event queue

      while (!eventQueue.empty() && eventQueue.front().time < lastStep) {
         // We shouldn't have previous epoch events in the queue
         assert(eventQueue.front().time >= firstStep);
         buffer.put(eventQueue.front());
         eventQueue.pop();
      }

      return buffer;
   }

   /// @brief Retrieves the clock tick size as defined in the input file
   /// @return The user defined clock tick size
   int getClockTickSize()
   {
      return clockTickSize_;
   }

   /// @brief Retrieve the clock tick unit as defined in the input file
   /// @return The user defined clock tick unit
   string getClockTickUnit()
   {
      return clockTickUnit_;
   }

   /// @brief  Peeks into the event at the front of the vertex queue
   /// @param  vertexId  The ID of the vertex
   /// @throws out_of_range, if vertexId is  not found in the map
   /// @return    The event at the front of the given vertex queue
   T queueFront(const VertexId_t &vertexId)
   {
      return eventsMap_.at(vertexId).front();
   }

   /// @brief  Removes the event at the front of the vertex queue
   /// @param  vertexId  The ID of the vertex
   /// @return true if the front of the queue was removed false otherwise.
   ///         Returns false if there are no events for the given vertex
   bool queuePop(const VertexId_t &vertexId)
   {
      try {
         eventsMap_.at(vertexId).pop();
         return true;
      } catch (std::out_of_range const &err) {
         return false;
      }
   }

   /// @brief  True if the event queue for the given vertex is empty, false otherwise
   /// @param vertexId  The vertexId
   /// @return True if the event queue is empty, false otherwise
   bool queueEmpty(const VertexId_t &vertexId)
   {
      try {
         return eventsMap_.at(vertexId).empty();
      } catch (std::out_of_range const &err) {
         // return true if there are no events for the given vertexId
         return true;
      }
   }

   /// @brief  Registers a property with the given name and a pointer to a member of the
   ///         Event class. The pointer to member is stored in a boost::variant type that
   ///         is later used to assign the input event data to the correct member variable
   /// @param propName  The name of the property as defined in the input file
   /// @param property  The pointer to member variable where the property should be stored
   /// @return    True if the property was successfully registered, false otherwise
   bool registerProperty(const string &propName, EventMemberPtr property)
   {
      // Check that we the propName is not an empty string
      if (propName.empty()) {
         LOG4CPLUS_ERROR(fileLogger_, "Property name should not be an empty string");
         return false;
      }

      // The compiler will check that only a type that is part of the variant can be
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

   // Clock tick size variables
   int clockTickSize_;
   string clockTickUnit_;

   log4cplus::Logger fileLogger_;      // For logging into a file
   log4cplus::Logger consoleLogger_;   // For logging to console

   bool getProperty(T &event, string propName, EventMemberPtr &eventMbrPtr,
                    const boost::property_tree::ptree &pTree)
   {
      switch (eventMbrPtr.which()) {
         // variant.which() returns a value between 0 and number of types - 1
         case static_cast<int>(PropertyType::INTEGER): {
            int T::*propPtr = get<int T::*>(eventMbrPtr);
            event.*propPtr = pTree.get<int>(propName);
         } break;
         case static_cast<int>(PropertyType::LONG): {
            long T::*propPtr = get<long T::*>(eventMbrPtr);
            event.*propPtr = pTree.get<long>(propName);
         } break;
         case static_cast<int>(PropertyType::UINT64): {
            uint64_t T::*propPtr = get<uint64_t T::*>(eventMbrPtr);
            event.*propPtr = pTree.get<uint64_t>(propName);
         } break;
         case static_cast<int>(PropertyType::FLOAT): {
            float T::*propPtr = get<float T::*>(eventMbrPtr);
            event.*propPtr = pTree.get<float>(propName);
         } break;
         case static_cast<int>(PropertyType::DOUBLE): {
            double T::*propPtr = get<double T::*>(eventMbrPtr);
            event.*propPtr = pTree.get<double>(propName);
         } break;
         case static_cast<int>(PropertyType::STRING): {
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
