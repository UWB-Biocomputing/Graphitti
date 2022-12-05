
#pragma once

// #include <log4cplus/loggingmacros.h>
#include <iostream>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/foreach.hpp>
#include <functional>

// #include "ParameterManager.h"

using namespace std;
using boost::property_tree::ptree;
struct Event
{
   // The vertexId where the input event happen
   uint32_t vertexId;
   
   // The start of the event since the beggining of
   // the simulation in timesteps matches g_simulationStep type
   uint64_t startTime;
   // The duration of the even in timesteps
   uint32_t duration;
   // Even location
   double x;
   double y;
   string type;
};


class InputManager {
public:
   InputManager() : events_() {
      // Get a copy of the file logger to use with log4cplus macros
      // fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));
      // LOG4CPLUS_DEBUG(fileLogger_, "\nInitializing InputManager");
   }

   bool readInputs() {
      ptree pt;

      ifstream inputFile;
      // TODO: Remove this after testing
      string inputFilePath = "../../Tools/Inputs/SPD_calls.xml";

      // Retrieve input file name from ParameterManager
      // string xpath = "//inputFile/text()";
      // if (!ParameterManager::getInstance().getStringByXpath(xpath, inputFilePath)) {
      //    cerr << "InputManager: Count not find XML Path: " << xpath << ".\n";
      //    return false;
      // }

      inputFile.open(inputFilePath.c_str());
      if (!inputFile.is_open()) {
         cerr << "InputManager: Failed to open file: " << inputFilePath << ".\n";
         return false;
      }

      boost::property_tree::xml_parser::read_xml(inputFile, pt);
      // LOG4CPLUS_DEBUG(fileLogger_, "\nInputManager: File loaded successfully into ptree.");
      // function<uint32_t(const ptree&, string)> uintFunc = bind(static_cast<uint32_t(*)(const ptree&,string)>(&getNodeValue<uint32_t>), this);
      // bind(static_cast<uint32_t(*)()>(&ptree::get_value<uint32_t>));
      BOOST_FOREACH(ptree::value_type const& v, pt.get_child("data")) {
         if (v.first == "event") {
            Event e;
            e.vertexId = getNodeValue<uint32_t>(v.second, "vertex_id"); // v.second.get<uint32_t>("vertex_id");
            e.startTime = v.second.get<uint64_t>("time");
            e.duration = v.second.get<uint64_t>("duration");
            e.x = v.second.get<double>("x");
            e.y = v.second.get<double>("y");
            e.type = v.second.get<string>("type");
            events_.push_back(e);
         }
      }
      return true;
   }

   template <typename T>
   T getNodeValue(const ptree& pt, string nodeName) {
      T value = pt.get<T>(nodeName);
      return value;
   }

   vector<Event>& getEvents() {
      return events_;
   }

private:
   vector<Event> events_;

   // log4cplus::Logger fileLogger_;
};