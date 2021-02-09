/**
 * @file LayoutFactory.h
 *
 * @ingroup Simulator/Layouts
 * 
 * @brief A factory class for creating Vertices objects.
 */

#pragma once

#include <map>
#include <memory>
#include <string>

#include "Global.h"
#include "Layout.h"

using namespace std;

class LayoutFactory {

public:
   ~LayoutFactory();

   static LayoutFactory *getInstance() {
      static LayoutFactory instance;
      return &instance;
   }

   // Invokes constructor for desired concrete class
   shared_ptr<Layout> createLayout(const string &className);

   /// Delete these methods because they can cause copy instances of the singleton when using threads.
   LayoutFactory(LayoutFactory const &) = delete;
   void operator=(LayoutFactory const &) = delete;

private:
   /// Constructor is private to keep a singleton instance of this class.
   LayoutFactory();

   /// Smart pointer to layout instance
   shared_ptr<Layout> layoutInstance;

   /// Defines function type for usage in internal map
   typedef Layout *(*CreateFunction)(void);

   /// Defines map between class name and corresponding ::Create() function.
   typedef map<string, CreateFunction> LayoutFunctionMap;

   /// Makes class-to-function map an internal factory member.
   LayoutFunctionMap createFunctions;

   /// Retrieves and invokes correct ::Create() function.
   Layout *invokeCreateFunction(const string &className);

   /// Register neuron class and it's create function to the factory.
   void registerClass(const string &className, CreateFunction function);
};
