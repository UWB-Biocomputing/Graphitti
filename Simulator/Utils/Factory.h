/**
 * @file Factory.h
 * 
 * @ingroup Simulator/Utils
 *
 * @brief A factory template for creating factory class for any subsystem that requires a 
 *       factory to create a (singleton) concrete class instance at runtime, based on a class name specified in the configuration file.
 *       
 */

#pragma once

#include <map>
#include <memory>
#include <string>

/**
 * The templated CreateFunctionMap allows each class to create a concrete instance of itself at runtime. 
 * This approach helps to keep the code organized and avoids circular dependencies that can make the code less error-prone.
 * 
 * How it avoids circular dependency?
 * If a map was passed from the base class with all the instance of its derived class, it would create a circular dependency. 
 * Meaning, that the derived class would need to reference the base class,  
 * and the base class would also need to reference the derived class for the sake of this map, leading to an infinite loop of references.
 * 
 * In addition, the templated CreateFunctionMap safeguards against creating a Factory code of an undesired type.
 * A static assert is used to check for any undesired type; thereby generating a compile-time error for any request to instantiate 
 * a template for a type that has not been explicitly defined.
 * 
*/

template <typename T> std::map<std::string, T *(*)(void)> CreateFunctionMap(T *placeholder);


/**
 * Factory is one of the important design pattern of Graphitti. It is developed in order to
 * create a singleton instance of each subsystem: Layout, Connections, Edges, Vertices, and Recorders. 
 * This design pattern enables the base class to create the correct instances of each of the derived classes, 
 * specified by simulation configuration information loaded at runtime.
 * 
 * The Factory class is templated to provide a pre-designed code skeleton for any subsystem that requires a 
 * to create a (singleton) concrete class instance at runtime. CreateFunctionMap is a templated helper method to Factory
 * that not only returns a map of the derived class name and derived class instance for the instantiated base class; but also help 
 * in guarding against instantiating a Factory against the base class that has not been explicitly added in CreateFunctionMap.
 *
*/
template <typename T> class Factory {
public:
   // Default destructor
   ~Factory() = default;

   /// Acts as constructor, returns the instance of singleton object for the instantiated type.
   static Factory &getInstance()
   {
      // A pointer of the type is passed to CreateFunctionMap rather than an object as it is more flexible.
      // i.e, few class type may be pure virtual or may not have a public constructor and destructor
      // leading to a compile time errors if we create an object fot it. However, it is guaranteed we can create a pointer to any class and pass it as a parameter.
      // In addition the pointer is used only to determine its type, hence we can pass a nullptr making it easier to maintain.
      T *placeholder = nullptr;
      static Factory instance(CreateFunctionMap<T>(placeholder));
      return instance;
   }

   // Invokes constructor for desired concrete class
   ///
   /// @param className class name.
   /// @return Unique pointer to an instance if className is found in
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
};

#include "CreateFunctionMap.cpp"
