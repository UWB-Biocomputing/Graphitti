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
RNGFactory::RNGFactory() {
	// register rng classes
	registerClass("MTRand", &MTRand::Create);
	registerClass("Norm", &Norm::Create);
}

RNGFactory::~RNGFactory() { createFunctions.clear(); }

///  Register rng class and its creation function to the factory.
///
///  @param  className  rng class name.
///  @param  Pointer to the class creation function.
void RNGFactory::registerClass(const std::string& className, CreateFunction function) {
	createFunctions[className] = function;
}


/// Creates concrete instance of the desired rng class.
MTRand* RNGFactory::createRNG(const std::string& className) { return invokeCreateFunction(className); }

/// Create an instance of the rng class using the static ::Create() method.
///
/// The calling method uses this retrieval mechanism in 
/// value assignment.
MTRand* RNGFactory::invokeCreateFunction(const std::string& className) {
	for (auto i = createFunctions.begin(); i != createFunctions.end(); ++i) {
		if (className == i->first) return i->second();
	}
	return nullptr;
}
