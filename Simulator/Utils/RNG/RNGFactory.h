/**
 * @file RNGFactory.h
 * 
 * @ingroup Simulator/Utils/RNG
 *
 * @brief A factory class for creating RNG objects.
 */

#pragma once

#include "Global.h"
#include "MTRand.h"
#include <map>
#include <memory>
#include <string>

using namespace std;

class RNGFactory {
public:
   ~RNGFactory();

   static RNGFactory &getInstance()
   {
      static RNGFactory instance;
      return instance;
   }

   // Invokes constructor for desired concrete class
   unique_ptr<MTRand> createRNG(const string &className);

   /// Delete copy and move methods to avoid copy instances of the singleton
   RNGFactory(const RNGFactory &rngFactory) = delete;
   RNGFactory &operator=(const RNGFactory &rngFactory) = delete;

   RNGFactory(RNGFactory &&rngFactory) = delete;
   RNGFactory &operator=(RNGFactory &&rngFactory) = delete;

private:
   /// Constructor is private to keep a singleton instance of this class.
   RNGFactory();

   /* Type definitions */
   /// Defines function type for usage in internal map
   typedef MTRand *(*CreateFunction)(void);

   /// Defines map between class name and corresponding ::Create() function.
   typedef map<string, CreateFunction> RNGFunctionMap;

   /// Makes class-to-function map an internal factory member.
   RNGFunctionMap createFunctions;

   /// Register rng class and it's create function to the factory.
   void registerClass(const string &className, CreateFunction function);
};
