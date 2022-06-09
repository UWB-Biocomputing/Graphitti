/**
 * @file VerticesFactory.cpp
 * 
 * @ingroup Simulator/Vertices
 *
 * @brief A factory class for creating Vertices objects.
 */

#include "VerticesFactory.h"
#include "All911Vertices.h"
#include "AllIZHNeurons.h"
#include "AllLIFNeurons.h"

/// Constructor is private to keep a singleton instance of this class.
VerticesFactory::VerticesFactory()
{
   // register vertices classes
   registerClass("AllLIFNeurons", &AllLIFNeurons::Create);
   registerClass("AllIZHNeurons", &AllIZHNeurons::Create);
   registerClass("All911Vertices", &All911Vertices::Create);
}

VerticesFactory::~VerticesFactory()
{
   createFunctions.clear();
}

///  Register vertices class and its creation function to the factory.
///
///  @param  className  Vertices class name.
///  @param  function   Pointer to the class creation function.
void VerticesFactory::registerClass(const string &className, CreateFunction function)
{
   createFunctions[className] = function;
}


/// Creates concrete instance of the desired vertices class.
///
/// @param className Vertices class name.
/// @return Shared pointer to vertices instance if className is found in
///         createFunctions map, nullptr otherwise.
shared_ptr<AllVertices> VerticesFactory::createVertices(const string &className)
{
   auto createVerticesIter = createFunctions.find(className);
   if (createVerticesIter != createFunctions.end()) {
      return shared_ptr<AllVertices>(createVerticesIter->second());
   }

   return nullptr;
}
