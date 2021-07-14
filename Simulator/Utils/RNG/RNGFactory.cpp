/**
 * @file RNGFactory.cpp
 * 
 * @ingroup Simulator/Utils/RNG
 *
 * @brief A factory class for creating RNG objects.
 */

#include "RNGFactory.h"
#include "MersenneTwister.h"
#include "Norm.h"

/// Constructor is private to keep a singleton instance of this class.
RNGFactory::RNGFactory() {
   // register vertices classes
   registerClass("MTRand", &MTRand::Create);
   registerClass("Norm", &Norm::Create);
}

RNGFactory::~RNGFactory() {
   createFunctions.clear();
}

///  Register vertices class and its creation function to the factory.
///
///  @param  className  vertices class name.
///  @param  Pointer to the class creation function.
void RNGFactory::registerClass(const string &className, CreateFunction function) {
   createFunctions[className] = function;
}


/// Creates concrete instance of the desired vertices class.
shared_ptr<MTRand> RNGFactory::createVertices(const string &className) {
   verticesInstance = shared_ptr<MTRand>(invokeCreateFunction(className));
   return verticesInstance;
}

/// Create an instance of the vertices class using the static ::Create() method.
///
/// The calling method uses this retrieval mechanism in 
/// value assignment.
MTRand *RNGFactory::invokeCreateFunction(const string &className) {
   for (auto i = createFunctions.begin(); i != createFunctions.end(); ++i) {
      if (className == i->first)
         return i->second();
   }
   return nullptr;
}
