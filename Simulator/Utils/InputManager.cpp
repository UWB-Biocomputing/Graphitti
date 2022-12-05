

#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/foreach.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/local_time/local_time.hpp>
#include <boost/property_map/dynamic_property_map.hpp>
#include <boost/property_map/property_map.hpp>
#include <iostream>
#include <utility>
#include <functional>
#include <boost/type_index.hpp>

#include "InputManager.h"

// enum EventType { EMS, Fire, Law  };
// boost::dynamic_properties properties(boost::ignore_other_properties);
// std::map<std::string, boost::type<boost::any>> getterMap;


// template <class Property>
// inline void registerProperty(const std::string &propName, Property property)
// {
//     std::map<std::string, Property> eventMap;
//     boost::associative_property_map<std::map<std::string, Property>> propMap(eventMap);
//     eventMap.insert(std::make_pair("event", property));
//     properties.property(propName, propMap);
// };

int main() {

//     std::map<std::string, double Event::*> eventDouble;
//     std::map<std::string, uint32_t Event::*> eventInt;

    // typedef typename std::map<std::string, double Event::*>::mapped_type dbl;
    
//     eventDouble.insert(std::make_pair("x", &Event::x));
//     eventDouble.insert(std::make_pair("y", &Event::y));
//     eventInt.insert(std::make_pair("id", &Event::vertexId));

//     Event newEvent;
//     double Event::*ptr = eventDouble.at("x");

//    std::cout << boost::typeindex::type_id_with_cvr<decltype(&Event::x)>().pretty_name() << "\n";
//    std::cout << boost::typeindex::type_id_with_cvr<dbl>().pretty_name() << "\n";

//     newEvent.*ptr = 10.5;
//     std::cout << "x from dynamic prop: " << newEvent.x << std::endl;


    // std::ifstream input_file;
    // input_file.open("../../Tools/Inputs/SPD_calls.xml");

    std:cerr << "Starting test.\n";

    InputManager inputManager;
    inputManager.readInputs();

    vector<Event> ie = inputManager.getEvents();
    for (auto e : ie) {
        std::cout << "x: " << e.x << " y: " << e.y << " vertexId: " << e.vertexId << std::endl;
    }
};