/**
 * @file CreateFunctionMap.cpp
 *
 * @ingroup Simulator/Utils
 *
 * @brief The templated CreateFunctionMap allows each class to create a concrete
 * instance of itself at runtime.
 *
 */

#include <map>
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

/**
 *
 * The templated CreateFunctionMap allows each class to create a concrete
 * instance of itself at runtime. This approach helps to keep the code organized
 * and avoids circular dependencies that can make the code less error-prone.
 *
 * How it avoids circular dependency?
 * If a map was passed from the base class with all the instance of its derived
 * class, it would create a circular dependency. Meaning, that the derived class
 * would need to reference the base class, and the base class would also need to
 * reference the derived class for the sake of this map, leading to an infinite
 * loop of references.
 *
 * In addition, the templated CreateFunctionMap safeguards against creating a
 * template of an undesired type. A static assert is used to check for any
 * undesired type; thereby generating a compile-time error for any request to
 * instantiate a template for a type that has not been explicitly defined.
 *
 * Why the T* argument?
 * C++ functions resolve polymorphic behavior based on their input arguments,
 * not their return types. This is because the compiler generates a separate
 * instance of the function for each function signature. For the templated
 * CreateFunctionMap, the T* argument is only used to resolve polymorphic
 * behavior. Also, a pointer is more flexible to pass as a parameter than an
 * object as the class may or may not have a public constructor and destructor.
 *
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

template <typename T> std::map<std::string, T *(*)(void)> CreateFunctionMap(T *placeholder)
{
   std::map<std::string, T *(*)(void)> createFunctionMap;

   // Check for undesired type
   static_assert((std::is_same_v<T, Connections> || std::is_same_v<T, AllEdges>
                  || std::is_same_v<T, Layout> || std::is_same_v<T, AllVertices>
                  || std::is_same_v<T, IRecorder> || std::is_same_v<T, MTRand>),
                 "Factory does not support instantiating this type");

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
