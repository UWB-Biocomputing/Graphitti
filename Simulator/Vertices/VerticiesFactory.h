/**
 * @file VerticesFactory.h
 * 
 * @ingroup Simulation/Vertices
 *
 * @brief A factory class for creating Vertices objects.
 */

#pragma once

#include <map>
#include <memory>
#include <string>

#include "Global.h"
#include "IAllVertices.h"

using namespace std;

class VerticesFactory {

public:
   ~VerticesFactory();

   static VerticesFactory *getInstance() {
      static VerticesFactory instance;
      return &instance;
   }

   // Invokes constructor for desired concrete class
   shared_ptr<IAllVertices> createVertices(const string &className);

   /// Delete these methods because they can cause copy instances of the singleton when using threads.
   VerticesFactory(VerticesFactory const &) = delete;
   void operator=(VerticesFactory const &) = delete;

private:
   /// Constructor is private to keep a singleton instance of this class.
   VerticesFactory();

   /// Pointer to vertices instance
   shared_ptr<IAllVertices> verticesInstance;

   /* Type definitions */
   /// Defines function type for usage in internal map
   typedef IAllVertices *(*CreateFunction)(void);

   /// Defines map between class name and corresponding ::Create() function.
   typedef map<string, CreateFunction> VerticesFunctionMap;

   /// Makes class-to-function map an internal factory member.
   VerticesFunctionMap createFunctions;

   /// Retrieves and invokes correct ::Create() function.
   IAllVertices *invokeCreateFunction(const string &className);

   /// Register vertex class and it's create function to the factory.
   void registerClass(const string &className, CreateFunction function);
};
