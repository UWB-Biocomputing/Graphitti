/**
 * @file RNGFactory.h
 * 
 * @ingroup Simulator/Utils/RNG
 *
 * @brief A factory class for creating RNG objects.
 */

#pragma once

#include <map>
#include <memory>
#include <string>

#include "Global.h"
#include "MTRand.h"


class RNGFactory {

	public:
		~RNGFactory();

		static RNGFactory* getInstance() {
			static RNGFactory instance;
			return &instance;
		}

		// Invokes constructor for desired concrete class
		MTRand* createRNG(const std::string& className);

		/// Delete these methods because they can cause copy instances of the singleton when using threads.
		RNGFactory(const RNGFactory&) = delete;
		void operator=(const RNGFactory&) = delete;

	private:
		/// Constructor is private to keep a singleton instance of this class.
		RNGFactory();

		/// Pointer to rng instance
		std::shared_ptr<MTRand> rngInstance;

		/* Type definitions */
		/// Defines function type for usage in internal map
		using CreateFunction = MTRand *(*)(void);

		/// Defines map between class name and corresponding ::Create() function.
		using RNGFunctionMap = std::map<std::string, CreateFunction>;

		/// Makes class-to-function map an internal factory member.
		RNGFunctionMap createFunctions;

		/// Retrieves and invokes correct ::Create() function.
		MTRand* invokeCreateFunction(const std::string& className);

		/// Register rng class and it's create function to the factory.
		void registerClass(const std::string& className, CreateFunction function);
};
