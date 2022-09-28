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
#include "AllSTDPSynapses.h"
#include "AllSpikingSynapses.h"

/// Constructor is private to keep a singleton instance of this class.
EdgesFactory::EdgesFactory()
{
   // register edges classes
   registerClass("AllSpikingSynapses", &AllSpikingSynapses::Create);
   registerClass("AllSTDPSynapses", &AllSTDPSynapses::Create);
   registerClass("AllDSSynapses", &AllDSSynapses::Create);
   registerClass("AllDynamicSTDPSynapses", &AllDynamicSTDPSynapses::Create);
   registerClass("All911Edges", &All911Edges::Create);
}

EdgesFactory::~EdgesFactory()
{
   createFunctions.clear();
}

///  Register edges class and its creation function to the factory.
///
///  @param  className  Edges class name.
///  @param  function   Pointer to the class creation function.
void EdgesFactory::registerClass(const string &className, CreateFunction function)
{
   createFunctions[className] = function;
}


/// Creates concrete instance of the desired edges class.
///
/// @param className Edges class name.
/// @return Shared pointer to edges intance if className is found in
///         createFunctions map, nullptr otherwise.
shared_ptr<AllEdges> EdgesFactory::createEdges(const string &className)
{
   auto createEdgesIter = createFunctions.find(className);
   if (createEdgesIter != createFunctions.end()) {
      return shared_ptr<AllEdges>(createEdgesIter->second());
   }

   return nullptr;
}
