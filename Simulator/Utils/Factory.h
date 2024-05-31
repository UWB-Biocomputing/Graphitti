/**
 * @file Factory.h
 *
 * @ingroup Simulator/Utils
 *
 * @brief A factory template for creating factory class for any subsystem that
 * requires a factory to create a (singleton) concrete class instance at
 * runtime. The class instance is based on a class name specified in the
 * configuration file.
 *
 * @details Factory is one of the important design patterns of Graphitti. It is
 * developed in order to create a singleton instance of each subsystem: Layout,
 * Connections, Edges, Vertices, and Recorders. This design pattern enables the
 * base class to create the correct instances of each of the derived classes,
 * specified by simulation configuration information file loaded at runtime.
 *
 *          The Factory class is templated to provide a pre-designed code
 * skeleton for any subsystem that requires a to create a (singleton) concrete
 * class instance at runtime. However, the factory class needs to have a map
 * that includes, for each concrete derived class, its name (string) and the
 * creation function for that particular class.
 *
 *          So, the obvious question is "which class populates this map, given
 * that it needs to be done before the factory can instantiate any object?"
 *
 *          One solution would be to pass a map from the respective base class
 * with all the instance of its derived class. Though simple, this solution
 * gives rise to a circular dependency; i.e, the derived class would need to
 * reference the base class, and the base class would also need to reference the
 * derived class for the sake of this map, leading to an infinite loop of
 * references.
 *
 *          Second solution - the choosen solution for this implementation, is
 * creating a static method in Factory template. This method- CreateFunctionMap,
 * returns a map with the required instance of derived classes for the
 * respective base class. This approach helps to keep the code organized and
 * avoids circular dependencies. However, the downside of this approach is that
 * the Factory.h file needs to include every definition of every class that
 * might be instantiated. This could be considered a plus- that all of the map
 * entries are organized in this one file.
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
#include "Layouts/Neuro/LayoutNeuro.h"

// Vertices
#include "Vertices/NG911/All911Vertices.h"
#include "Vertices/Neuro/AllIZHNeurons.h"
#include "Vertices/Neuro/AllLIFNeurons.h"

// Recorder
#include "Recorders/NG911/Xml911Recorder.h"
#include "Recorders/Recorder.h"
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

   /// Acts as constructor, returns the instance of singleton object for the
   /// instantiated type.
   static Factory &getInstance()
   {
      static Factory instance(CreateFunctionMap());
      return instance;
   }

   // Invokes constructor for desired concrete class
   ///
   /// @param className class name.
   /// @return Unique pointer to an instance if className is found in
   ///         createFunctionsMap map, nullptr otherwise.
   ///
   std::unique_ptr<T> createType(const std::string &className)
   {
      auto createIter = createFunctionsMap_.find(className);
      if (createIter != createFunctionsMap_.end()) {
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
   FunctionMap createFunctionsMap_;

   /// Constructor is private to keep a singleton instance of this class.
   Factory(std::map<std::string, CreateFunction> map)
   {
      createFunctionsMap_ = std::move(map);
   }

   /// @brief     A static method that returns a map with the required instance
   /// of
   ///            derived classes for the respective base class.
   /// @return    Returns a map with the name (string) of each concrete derived
   /// class,
   ///            and its creation function.
   ///
   static std::map<std::string, CreateFunction> CreateFunctionMap()
   {
      std::map<std::string, CreateFunction> createFunctionMap;

      // A static assert is used to check for any undesired type; thereby
      // generating a compile-time error for any request to instantiate a template
      // for a type that has not been explicitly defined.
      static_assert((std::is_same_v<T, Connections> || std::is_same_v<T, AllEdges>)
                       || (std::is_same_v<T, Layout> || std::is_same_v<T, AllVertices>)
                       || (std::is_same_v<T, Recorder> || std::is_same_v<T, MTRand>),
                    "Invalid object type passed to CreateFunctionMap");

      //  What is std::is_same<> ?
      //  std::is_same<> is a type trait in C++ that checks whether two types are
      //  the same or not. It is a compile-time type trait that returns a bool
      //  value indicating whether the two types are the same.

      //  Why can std::is_same<> be constexpr?
      //  The std::is_same<> can be evaluated at compile-time because they are
      //  implemented as constexpr functions. This is possible because
      //  std::is_same<> does not actually create any objects or perform any
      //  operations at runtime, it simply checks the types of the given template
      //  arguments and returns a boolean value.

      // Register Connections classes
      if constexpr (std::is_same_v<T, Connections>) {
         createFunctionMap["ConnStatic"] = &ConnStatic::Create;
         createFunctionMap["ConnGrowth"] = &ConnGrowth::Create;
         createFunctionMap["Connections911"] = &Connections911::Create;
      }

      // Register AllEdges classes
      else if constexpr (std::is_same_v<T, AllEdges>) {
         createFunctionMap["All911Edges"] = &All911Edges::Create;
         createFunctionMap["AllDSSynapses"] = &AllDSSynapses::Create;
         createFunctionMap["AllSTDPSynapses"] = &AllSTDPSynapses::Create;
         createFunctionMap["AllSpikingSynapses"] = &AllSpikingSynapses::Create;
         createFunctionMap["AllDynamicSTDPSynapses"] = &AllDynamicSTDPSynapses::Create;
      }

      // Register Layout classes
      else if constexpr (std::is_same_v<T, Layout>) {
         createFunctionMap["Layout911"] = &Layout911::Create;
         createFunctionMap["LayoutNeuro"] = &LayoutNeuro::Create;
      }

      // Register AllVertices classes
      else if constexpr (std::is_same_v<T, AllVertices>) {
         createFunctionMap["AllLIFNeurons"] = &AllLIFNeurons::Create;
         createFunctionMap["AllIZHNeurons"] = &AllIZHNeurons::Create;
         createFunctionMap["All911Vertices"] = &All911Vertices::Create;
      }

      // Register Recorder classes
      else if constexpr (std::is_same_v<T, Recorder>) {
         createFunctionMap["XmlRecorder"] = &XmlRecorder::Create;
         createFunctionMap["Xml911Recorder"] = &Xml911Recorder::Create;

#if defined(HDF5)
         createFunctionMap["Hdf5Recorder"] = &Hdf5Recorder::Create;
         createFunctionMap["Hdf5GrowthRecorder"] = &Hdf5GrowthRecorder::Create;
#endif
      }

      // Register MTRand classes
      else if constexpr (std::is_same_v<T, MTRand>) {
         createFunctionMap["Norm"] = &Norm::Create;
         createFunctionMap["MTRand"] = &MTRand::Create;
      }

      return createFunctionMap;
   }
};
