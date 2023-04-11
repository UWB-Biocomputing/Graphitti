/**
 * @file LayoutFactory.h
 *
 * @ingroup Simulator/Layouts
 * 
 * @brief A factory class for creating Vertices objects.
 */

#pragma once

#include "Global.h"
#include "Layout.h"
#include <map>
#include <memory>
#include <string>

using namespace std;

class LayoutFactory {
public:
   static LayoutFactory &getInstance()
   {
      static LayoutFactory instance;
      return instance;
   }

   // Invokes constructor for desired concrete class
   unique_ptr<Layout> createLayout(const string &className);

   /// Delete copy and move methods to avoid copy instances of the singleton
   LayoutFactory(const LayoutFactory &layoutFactory) = delete;
   LayoutFactory &operator=(const LayoutFactory &layoutFactory) = delete;

   LayoutFactory(LayoutFactory &&layoutFactory) = delete;
   LayoutFactory &operator=(LayoutFactory &&layoutFactory) = delete;

private:
   /// Constructor is private to keep a singleton instance of this class.
   LayoutFactory();

   /// Defines function type for usage in internal map
   typedef Layout *(*CreateFunction)(void);

   /// Defines map between class name and corresponding ::Create() function.
   typedef map<string, CreateFunction> LayoutFunctionMap;

   /// Makes class-to-function map an internal factory member.
   LayoutFunctionMap createFunctions;

   /// Register neuron class and it's create function to the factory.
   void registerClass(const string &className, CreateFunction function);
};
