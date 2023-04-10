/**
 * @file ConnectionsFactory.cpp
 * 
 * @ingroup Simulator/Connections
 *
 * @brief A factory class for creating Connections objects.
 */

#include "ConnectionsFactory.h"
#include "ConnGrowth.h"
#include "ConnStatic.h"
#include "Connections911.h"


/// Constructor is private to keep a singleton instance of this class.
ConnectionsFactory::ConnectionsFactory()
{
   // register connections classes
   registerClass("ConnStatic", &ConnStatic::Create);
   registerClass("ConnGrowth", &ConnGrowth::Create);
   registerClass("Connections911", &Connections911::Create);
}

ConnectionsFactory::~ConnectionsFactory()
{
   createFunctions.clear();
}


///  Register connections class and its creation function to the factory.
///
///  @param  className  Class name.
///  @param  function   Pointer to the class creation function.
void ConnectionsFactory::registerClass(const string &className, CreateFunction function)
{
   createFunctions[className] = function;
}


/// Creates concrete instance of the desired connections class.
///
/// @param className Connections class name.
/// @return Shared pointer to connections instance if className is found in
///         createFunctions map, nullptr otherwise.
unique_ptr<Connections> ConnectionsFactory::createConnections(const string &className)
{
   auto createConnectionsIter = createFunctions.find(className);
   if (createConnectionsIter != createFunctions.end()) {
      return unique_ptr<Connections>(createConnectionsIter->second());
   }

   return nullptr;
}
