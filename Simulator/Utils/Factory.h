/**
 * @file Factory.h
 * 
 * @ingroup Simulator/Utils
 *
 * @brief A factory template for creating factory class for 
 *        Connections, Edges, Layout, Verticies, Recorder and MTRand.
 */

#pragma once

#include <map>
#include <memory>
#include <string>

template <typename T> std::map<std::string, T *(*)(void)> getCreateFunctionForType();

template <typename T> class Factory {
public:
   ~Factory() = default;

   static Factory &getInstance()
   {
      static Factory instance(getCreateFunctionForType<T>());
      return instance;
   }

   // Invokes constructor for desired concrete class
   ///
   /// @param className class name.
   /// @return Shared pointer to an instance if className is found in
   ///         createFunctions map, nullptr otherwise.
   std::unique_ptr<T> createType(const std::string &className)
   {
      auto createIter = createFunctions.find(className);
      if (createIter != createFunctions.end()) {
         return std::unique_ptr<T>(createIter->second());
      }
      return nullptr;
   }

   /// Delete copy and move methods to avoid copy instances of the singleton
   Factory(const Factory &factory) = delete;
   Factory &operator=(const Factory &factory) = delete;

   Factory(Factory &&factory) = delete;
   Factory &operator=(Factory &&factory) = delete;

private:
   /// Defines function type for usage in internal map
   using CreateFunction = T *(*)(void);

   /// Defines map between class name and corresponding ::Create() function.
   using FunctionMap = std::map<std::string, CreateFunction>;

   /// Makes class-to-function map an internal factory member.
   FunctionMap createFunctions;

   /// Constructor is private to keep a singleton instance of this class.
   Factory(std::map<std::string, CreateFunction> map)
   {
      createFunctions = std::move(map);
   }

   /// Register class and it's create function to the factory.
   ///
   ///  @param  className  class name.
   ///  @param  function   Pointer to the class creation function.
   void registerClass(const std::string &className, CreateFunction function)
   {
      createFunctions[className] = function;
   }
};
