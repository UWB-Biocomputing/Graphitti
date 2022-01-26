/**
 * @file EdgesFactory.cpp
 * 
 * @ingroup Simulator/Edges
 *
 * @brief A factory class for creating Edges objects.
 */

#include "EdgesFactory.h"

#include "All911Edges.h"
#include "AllDSSynapses.h"
#include "AllDynamicSTDPSynapses.h"
#include "AllSpikingSynapses.h"
#include "AllSTDPSynapses.h"

/// Constructor is private to keep a singleton instance of this class.
EdgesFactory::EdgesFactory() {
	// register neurons classes
	registerClass("AllSpikingSynapses", &AllSpikingSynapses::Create);
	registerClass("AllSTDPSynapses", &AllSTDPSynapses::Create);
	registerClass("AllDSSynapses", &AllDSSynapses::Create);
	registerClass("AllDynamicSTDPSynapses", &AllDynamicSTDPSynapses::Create);
	registerClass("All911Edges", &All911Edges::Create);
}

EdgesFactory::~EdgesFactory() { createFunctions.clear(); }

///  Register edges class and its creation function to the factory.
///
///  @param  className  neurons class name.
///  @param  Pointer to the class creation function.
void EdgesFactory::registerClass(const std::string& className, CreateFunction function) {
	createFunctions[className] = function;
}


/// Creates concrete instance of the desired neurons class.
std::shared_ptr<AllEdges> EdgesFactory::createEdges(const std::string& className) {
	edgesInstance_ = std::shared_ptr<AllEdges>(invokeCreateFunction(className));
	return edgesInstance_;
}

/// Create an instance of the edges class using the static ::Create() method.
///
/// The calling method uses this retrieval mechanism in
/// value assignment.
AllEdges* EdgesFactory::invokeCreateFunction(const std::string& className) {
	for (auto i = createFunctions.begin(); i != createFunctions.end(); ++i) if (className == i->first) return i->
		second();
	return nullptr;
}
