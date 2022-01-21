/**
 * @file VerticesFactory.h
 * 
 * @ingroup Simulator/Vertices
 *
 * @brief A factory class for creating Vertices objects.
 */

#pragma once

#include <map>
#include <memory>
#include <string>

#include "Global.h"
#include "AllVertices.h"

 

class VerticesFactory {

public:
   ~VerticesFactory();

   static VerticesFactory *getInstance() {
      static VerticesFactory instance;
      return &instance;
   }

   // Invokes constructor for desired concrete class
   std::shared_ptr<AllVertices> createVertices(const std::string &className);

   /// Delete these methods because they can cause copy instances of the singleton when using threads.
   VerticesFactory(VerticesFactory const &) = delete;
   void operator=(VerticesFactory const &) = delete;

private:
   /// Constructor is private to keep a singleton instance of this class.
   VerticesFactory();

   /// Pointer to vertices instance
   std::shared_ptr<AllVertices> verticesInstance;

   /* Type definitions */
   /// Defines function type for usage in internal map
   using CreateFunction =  AllVertices *(*)();

   /// Defines map between class name and corresponding ::Create() function.
   using VerticesFunctionMap =  std::map<std::string, CreateFunction>;

   /// Makes class-to-function map an internal factory member.
   VerticesFunctionMap createFunctions;

   /// Retrieves and invokes correct ::Create() function.
   AllVertices *invokeCreateFunction(const std::string &className);

   /// Register vertex class and it's create function to the factory.
   void registerClass(const std::string &className, CreateFunction function);
};
