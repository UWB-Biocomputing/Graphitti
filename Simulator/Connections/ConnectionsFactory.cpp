/**
 * @file ConnectionsFactory.cpp
 * 
 * @ingroup Simulator/Connections
 *
 * @brief A factory class for creating Connections objects.
 */

#include "ConnectionsFactory.h"
#include "Connections911.h"
#include "ConnGrowth.h"
#include "ConnStatic.h"


/// Constructor is private to keep a singleton instance of this class.
ConnectionsFactory::ConnectionsFactory() {
	// register neurons classes
	registerClass("ConnStatic", &ConnStatic::Create);
	registerClass("ConnGrowth", &ConnGrowth::Create);
	registerClass("Connections911", &Connections911::Create);
}

ConnectionsFactory::~ConnectionsFactory() { createFunctions.clear(); }


///  Register connections class and its creation function to the factory.
///
///  @param  className class name.
///  @param  Pointer to the class creation function.
void ConnectionsFactory::registerClass(const std::string& className, CreateFunction function) {
	createFunctions[className] = function;
}


/// Creates concrete instance of the desired connections class.
std::shared_ptr<Connections> ConnectionsFactory::createConnections(const std::string& className) {
	connectionsInstance = std::shared_ptr<Connections>(invokeCreateFunction(className));
	return connectionsInstance;
}

/// Create an instance of the connections class using the static ::Create() method.
///
/// The calling method uses this retrieval mechanism in
/// value assignment.
Connections* ConnectionsFactory::invokeCreateFunction(const std::string& className) {
	for (auto i = createFunctions.begin(); i != createFunctions.end(); ++i) {
		if (className == i->first) return i->second();
	}
	return nullptr;
}
