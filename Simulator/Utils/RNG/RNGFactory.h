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
#include "MersenneTwister.h"

using namespace std;

class RNGFactory {

public:
   ~RNGFactory();

   static RNGFactory *getInstance() {
      static RNGFactory instance;
      return &instance;
   }

   // Invokes constructor for desired concrete class
   shared_ptr<MTRand> createRNG(const string &className);

   /// Delete these methods because they can cause copy instances of the singleton when using threads.
   RNGFactory(RNGFactory const &) = delete;
   void operator=(RNGFactory const &) = delete;

private:
   /// Constructor is private to keep a singleton instance of this class.
   RNGFactory();

   /// Pointer to rng instance
   shared_ptr<MTRand> rngInstance;

   /* Type definitions */
   /// Defines function type for usage in internal map
   typedef MTRand *(*CreateFunction)(void);

   /// Defines map between class name and corresponding ::Create() function.
   typedef map<string, CreateFunction> RNGFunctionMap;

   /// Makes class-to-function map an internal factory member.
   RNGFunctionMap createFunctions;

   /// Retrieves and invokes correct ::Create() function.
   MTRand *invokeCreateFunction(const string &className);

   /// Register rng class and it's create function to the factory.
   void registerClass(const string &className, CreateFunction function);
};
