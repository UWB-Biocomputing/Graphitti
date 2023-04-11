/**
 * @file RNGFactory.cpp
 * 
 * @ingroup Simulator/Utils/RNG
 *
 * @brief A factory class for creating RNG objects.
 */

#include "RNGFactory.h"
#include "MTRand.h"
#include "Norm.h"

/// Constructor is private to keep a singleton instance of this class.
RNGFactory::RNGFactory()
{
   // register rng classes
   registerClass("MTRand", &MTRand::Create);
   registerClass("Norm", &Norm::Create);
}

RNGFactory::~RNGFactory()
{
   createFunctions.clear();
}

///  Register rng class and its creation function to the factory.
///
///  @param  className  rng class name.
///  @param  Pointer to the class creation function.
void RNGFactory::registerClass(const string &className, CreateFunction function)
{
   createFunctions[className] = function;
}


/// Creates concrete instance of the desired rng class.
///
/// @param className rng class name.
/// @return Shared pointer to RNG instance if className is found in
///         createFunctions map, nullptr otherwise.
unique_ptr<MTRand> RNGFactory::createRNG(const string &className)
{
   auto createRNGIter = createFunctions.find(className);
   if (createRNGIter != createFunctions.end()) {
      return unique_ptr<MTRand>(createRNGIter->second());
   }
   return nullptr;
}
