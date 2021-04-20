/**
 * @file EdgesFactory.h
 * 
 * @ingroup Simulator/Edges
 *
 * @brief A factory class for creating Edges objects.
 */

#pragma once

#include <map>
#include <memory>
#include <string>

#include "Global.h"
#include "AllEdges.h"

using namespace std;

class EdgesFactory {

public:
   ~EdgesFactory();

   static EdgesFactory *getInstance() {
      static EdgesFactory instance;
      return &instance;
   }

   // Invokes constructor for desired concrete class
   shared_ptr<AllEdges> createEdges(const string &className);

   // Delete these methods because they can cause copy instances of the singleton when using threads.
   EdgesFactory(EdgesFactory const &) = delete;
   void operator=(EdgesFactory const &) = delete;

private:
   /// Constructor is private to keep a singleton instance of this class.
   EdgesFactory();

   /// Pointer to edges instance
   shared_ptr<AllEdges> edgesInstance_;

   /// Defines function type for usage in internal map
   typedef AllEdges *(*CreateFunction)(void);

   /// Defines map between class name and corresponding ::Create() function.
   typedef map<string, CreateFunction> EdgesFunctionMap;

   /// Makes class-to-function map an internal factory member.
   EdgesFunctionMap createFunctions;

   /// Retrieves and invokes correct ::Create() function.
   AllEdges *invokeCreateFunction(const string &className);

   /// Register edges class and it's create function to the factory.
   void registerClass(const string &className, CreateFunction function);
};
