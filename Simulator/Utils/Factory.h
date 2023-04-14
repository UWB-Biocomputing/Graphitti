/**
 * @file Factory.h
 * 
 * @ingroup Simulator/Utils
 *
 * @brief A factory template for creating factory class for any subsystem that requires a
 *        factory to create a (singleton) concrete class instance at runtime.
 *        The class instance is based on a class name specified in the configuration file.
 * 
 * @details Factory is one of the important design patterns of Graphitti. It is developed 
 *          in order to create a singleton instance of each subsystem: Layout, Connections,
 *          Edges, Vertices, and Recorders. This design pattern enables the base class to 
 *          create the correct instances of each of the derived classes, specified by 
 *          simulation configuration information file loaded at runtime.
 * 
 *          The Factory class is templated to provide a pre-designed code skeleton for any
 *          subsystem that requires a to create a (singleton) concrete class instance at
 *          runtime. However, the factory class needs to have a map that includes, for each
 *          concrete derived class, its name (string) and the creation function for that 
 *          particular class.    
 * 
 *          So, the obvious question is "which class populates this map, given that it needs
 *          to be done before the factory can instantiate any object?" 
 * 
 *          One solution would be to pass a map from the respective base class with all the
 *          instance of its derived class. Though simple, this solution gives rise to a 
 *          circular dependency. i.e, the derived class would need to reference the base class,
 *          and the base class would also need to reference the derived class for the sake of 
 *          this map, leading to an infinite loop of references.
 * 
 *          Second solution - the choosen solution for this implementation, is creating a static
 *          method in Factory template. This method- CreateFunctionMap, returns a map with the 
 *          required instance of derived classes for the respective base class. This approach 
 *          helps to keep the code organized and avoids circular dependencies. However, the  
 *          downside of this approach is that the Factory.h file needs to include every 
 *          definition of every class that might be instantiated. This could be considered a 
 *          plus- that all of the map entries are organized in this one file.
 * 
*/

#pragma once

#include <map>
#include <memory>
#include <string>

// Connections
#include "Connections/NG911/Connections911.h"
#include "Connections/Neuro/ConnGrowth.h"
#include "Connections/Neuro/ConnStatic.h"

// Edges
#include "Edges/NG911/All911Edges.h"
#include "Edges/Neuro/AllDSSynapses.h"
#include "Edges/Neuro/AllDynamicSTDPSynapses.h"
#include "Edges/Neuro/AllSTDPSynapses.h"
#include "Edges/Neuro/AllSpikingSynapses.h"

// Layout
#include "Layouts/NG911/Layout911.h"
#include "Layouts/Neuro/DynamicLayout.h"
#include "Layouts/Neuro/FixedLayout.h"

// Vertices
#include "Vertices/NG911/All911Vertices.h"
#include "Vertices/Neuro/AllIZHNeurons.h"
#include "Vertices/Neuro/AllLIFNeurons.h"

// Recorder
#include "Recorders/IRecorder.h"
#include "Recorders/NG911/Xml911Recorder.h"
#include "Recorders/Neuro/XmlGrowthRecorder.h"
#include "Recorders/Neuro/XmlSTDPRecorder.h"
#include "Recorders/XmlRecorder.h"

#if defined(HDF5)
   #include "Recorders/Hdf5Recorder.h"
   #include "Recorders/Neuro/Hdf5GrowthRecorder.h"
#endif

// MTRand
#include "RNG/MTRand.h"
#include "RNG/Norm.h"

template <typename T> class Factory {
public:
   // Default destructor
   ~Factory() = default;

   /// Acts as constructor, returns the instance of singleton object for the instantiated type.
   static Factory &getInstance()
   {
      /** 
       * Why the T* argument?
       * C++ functions resolve polymorphic behavior based on their input arguments,
       * not their return types. This is because the compiler generates a separate
       * instance of the function for each function signature. 
       * 
       * For the CreateFunctionMap, the T* argument is only used to resolve polymorphic behavior.
       * Also, you can create a pointer, T*, without having to create an instance of the class T. 
       * In this case,it is essential, because the function expects a pointer to the (abstract) 
       * base class.
       */

      T *placeholder = nullptr;
      static Factory instance(CreateFunctionMap(placeholder));
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

   static std::map<std::string, T *(*)(void)> CreateFunctionMap(T *placeholder)
   {
      std::map<std::string, T *(*)(void)> createFunctionMap;

      // A static assert is used to check for any undesired type; thereby generating a
      // compile-time error for any request to instantiate a template for a type that
      // has not been explicitly defined.

      static_assert((std::is_same_v<T, Connections> || std::is_same_v<T, AllEdges>
                     || std::is_same_v<T, Layout> || std::is_same_v<T, AllVertices>
                     || std::is_same_v<T, IRecorder> || std::is_same_v<T, MTRand>),
                    "Invalid object type passed to CreateFunctionMap");

      /*
      * What is std::is_same<> doing?
      * std::is_same<> is a type trait in C++ that checks whether two types are the
      * same or not. It is a compile-time type trait that returns a bool value
      * indicating whether the two types are the same.
      *
      * Why can they be constexpr?
      * The std::is_same<> tests can be evaluated at compile-time because they are
      * implemented as constexpr functions. This is possible because std::is_same<>
      * does not actually create any objects or perform any operations at runtime, it
      * simply checks the types of the given template arguments and returns a boolean
      * value.
      *
      */

      // Register Connections classes
      if constexpr (std::is_same_v<T, Connections>) {
         createFunctionMap["ConnStatic"] = &ConnStatic::Create;
         createFunctionMap["ConnGrowth"] = &ConnGrowth::Create;
         createFunctionMap["Connections911"] = &Connections911::Create;
      }

      // Register AllEdges classes
      else if constexpr (std::is_same_v<T, AllEdges>) {
         createFunctionMap["AllSpikingSynapses"] = &AllSpikingSynapses::Create;
         createFunctionMap["AllSTDPSynapses"] = &AllSTDPSynapses::Create;
         createFunctionMap["AllDSSynapses"] = &AllDSSynapses::Create;
         createFunctionMap["AllDynamicSTDPSynapses"] = &AllDynamicSTDPSynapses::Create;
         createFunctionMap["All911Edges"] = &All911Edges::Create;
      }

      // Register Layout classes
      else if constexpr (std::is_same_v<T, Layout>) {
         createFunctionMap["FixedLayout"] = &FixedLayout::Create;
         createFunctionMap["DynamicLayout"] = &DynamicLayout::Create;
         createFunctionMap["Layout911"] = &Layout911::Create;
      }

      // Register AllVertices classes
      else if constexpr (std::is_same_v<T, AllVertices>) {
         createFunctionMap["AllLIFNeurons"] = &AllLIFNeurons::Create;
         createFunctionMap["AllIZHNeurons"] = &AllIZHNeurons::Create;
         createFunctionMap["All911Vertices"] = &All911Vertices::Create;
      }

      // Register IRecorder classes
      else if constexpr (std::is_same_v<T, IRecorder>) {
         createFunctionMap["XmlRecorder"] = &XmlRecorder::Create;
         createFunctionMap["XmlGrowthRecorder"] = &XmlGrowthRecorder::Create;
         createFunctionMap["XmlSTDPRecorder"] = &XmlSTDPRecorder::Create;
         createFunctionMap["Xml911Recorder"] = &Xml911Recorder::Create;
#if defined(HDF5)
         createFunctionMap["Hdf5Recorder"] = &Hdf5Recorder::Create;
         createFunctionMap["Hdf5GrowthRecorder"] = &Hdf5GrowthRecorder::Create;
#endif
      }

      // Register MTRand classes
      else if constexpr (std::is_same_v<T, MTRand>) {
         createFunctionMap["MTRand"] = &MTRand::Create;
         createFunctionMap["Norm"] = &Norm::Create;
      }

      return createFunctionMap;
   }
};
