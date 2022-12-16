
#include "InputManager.h"
#include <boost/foreach.hpp>
#include <boost/property_tree/exceptions.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <iostream>

using boost::property_tree::ptree;

InputManager::InputManager()
{
   // Get a copy of the file logger to use with log4cplus macros
   fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));
   LOG4CPLUS_DEBUG(fileLogger_, "Initializing InputManager");
}

bool InputManager::readInputs()
{
   ptree pt;
   ifstream inputFile;
   string inputFilePath;

   // Retrieve input file name from ParameterManager
   string xpath = "//inputFile/text()";
   if (!ParameterManager::getInstance().getStringByXpath(xpath, inputFilePath)) {
      cerr << "InputManager: Count not find XML Path: " << xpath << ".\n";
      return false;
   }

   inputFile.open(inputFilePath.c_str());
   if (!inputFile.is_open()) {
      cerr << "InputManager: Failed to open file: " << inputFilePath << ".\n";
      return false;
   }

   boost::property_tree::xml_parser::read_xml(inputFile, pt);
   BOOST_FOREACH (ptree::value_type const &v, pt.get_child("data")) {
      if (v.first == "event") {
         try {
            // We will only read registered attributes
            Event event;   // the event object to record
            for (auto propPair : registeredPropMap_) {
               getProperty(event, propPair.first, propPair.second, v.second);
            }

            // operator[] inserts a new element with that key if it doesn't already
            // exists in the map
            eventsMap_[event.vertexId].push(event);
         } catch (boost::property_tree::ptree_bad_data e) {
            LOG4CPLUS_FATAL(fileLogger_, "InputManager failed to read event node: " << e.what());
            // TODO: perhaps we need to exit since there is missing info here
            return false;
         } catch (boost::bad_get e) {
            LOG4CPLUS_FATAL(fileLogger_, "Failed to read event property: " << e.what());
            return false;
         }
      }
   }
   LOG4CPLUS_DEBUG(fileLogger_, "Input file loaded successfully");
   return true;
}

Event InputManager::vertexQueueFront(const VertexId_t &vertexId)
{
   // TODO: this throws an out_of_range exception if vertexId is not found
   return eventsMap_.at(vertexId).front();
}

void InputManager::vertexQueuePop(const VertexId_t &vertexId)
{
   // TODO: handle when vertexID is not found in the map
   eventsMap_.at(vertexId).pop();
}

bool InputManager::registerProperty(const string &propName, EventMemberPtr property)
{
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

vector<Event> InputManager::getEvents(const VertexId_t &vertexId, uint64_t firstStep, uint64_t lastStep) {
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

bool InputManager::getProperty(Event &event, string propName, EventMemberPtr &eventMbrPtr,
                               const ptree &pTree)
{
   switch (eventMbrPtr.which()) {
      // variant.which() returns a value between 0 and number of types - 1
      case PropertyType::INTEGER: {
         int Event::*propPtr = get<int Event::*>(eventMbrPtr);
         event.*propPtr = pTree.get<int>(propName);
      } break;
      case PropertyType::LONG: {
         long Event::*propPtr = get<long Event::*>(eventMbrPtr);
         event.*propPtr = pTree.get<long>(propName);
      } break;
      case PropertyType::UINT64: {
         uint64_t Event::*propPtr = get<uint64_t Event::*>(eventMbrPtr);
         event.*propPtr = pTree.get<uint64_t>(propName);
      } break;
      case PropertyType::FLOAT: {
         float Event::*propPtr = get<float Event::*>(eventMbrPtr);
         event.*propPtr = pTree.get<float>(propName);
      } break;
      case PropertyType::DOUBLE: {
         double Event::*propPtr = get<double Event::*>(eventMbrPtr);
         event.*propPtr = pTree.get<double>(propName);
      } break;
      case PropertyType::STRING: {
         string Event::*propPtr = get<string Event::*>(eventMbrPtr);
         event.*propPtr = pTree.get<string>(propName);
      } break;
      default:
         LOG4CPLUS_DEBUG(fileLogger_, "Property not supported: " << propName);
         return false;
   }
   return true;
}