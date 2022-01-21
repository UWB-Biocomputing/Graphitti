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

class EdgesFactory {

public:
   ~EdgesFactory();

   static EdgesFactory *getInstance() {
      static EdgesFactory instance;
      return &instance;
   }

   // Invokes constructor for desired concrete class
   std::shared_ptr<AllEdges> createEdges(const std::string &className);

   // Delete these methods because they can cause copy instances of the singleton when using threads.
   EdgesFactory(EdgesFactory const &) = delete;
   void operator=(EdgesFactory const &) = delete;

private:
   /// Constructor is private to keep a singleton instance of this class.
   EdgesFactory();

   /// Pointer to edges instance
   std::shared_ptr<AllEdges> edgesInstance_;

   /// Defines function type for usage in internal map
   using CreateFunction =  AllEdges *(*)(void);

   /// Defines map between class name and corresponding ::Create() function.
   using EdgesFunctionMap =  std::map<std::string, CreateFunction>;

   /// Makes class-to-function map an internal factory member.
   EdgesFunctionMap createFunctions;

   /// Retrieves and invokes correct ::Create() function.
   AllEdges *invokeCreateFunction(const std::string &className);

   /// Register edges class and it's create function to the factory.
   void registerClass(const std::string &className, CreateFunction function);
};
