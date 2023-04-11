/**
 * @file Factory.cpp
 * 
 * @ingroup Simulator/Utils
 *
 * @brief A factory template for creating factory class for 
 *        Connections, Edges, Layout, Verticies, Recorder and MTRand.
 */

// #include "Factory.h"
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

//MTRand
#include "RNG/MTRand.h"
#include "RNG/Norm.h"

template <typename T> std::map<std::string, T *(*)(void)> getCreateFunctionForType(T *placeholder)
{
   std::map<std::string, T *(*)(void)> createFunctionMap;

   if constexpr (std::is_same_v<T, Connections>) {
      createFunctionMap["ConnStatic"] = &ConnStatic::Create;
      createFunctionMap["ConnGrowth"] = &ConnGrowth::Create;
      createFunctionMap["Connections911"] = &Connections911::Create;
   } else if constexpr (std::is_same_v<T, AllEdges>) {
      createFunctionMap["AllSpikingSynapses"] = &AllSpikingSynapses::Create;
      createFunctionMap["AllSTDPSynapses"] = &AllSTDPSynapses::Create;
      createFunctionMap["AllDSSynapses"] = &AllDSSynapses::Create;
      createFunctionMap["AllDynamicSTDPSynapses"] = &AllDynamicSTDPSynapses::Create;
      createFunctionMap["All911Edges"] = &All911Edges::Create;
   } else if constexpr (std::is_same_v<T, Layout>) {
      createFunctionMap["FixedLayout"] = &FixedLayout::Create;
      createFunctionMap["DynamicLayout"] = &DynamicLayout::Create;
      createFunctionMap["Layout911"] = &Layout911::Create;
   } else if constexpr (std::is_same_v<T, AllVertices>) {
      createFunctionMap["AllLIFNeurons"] = &AllLIFNeurons::Create;
      createFunctionMap["AllIZHNeurons"] = &AllIZHNeurons::Create;
      createFunctionMap["All911Vertices"] = &All911Vertices::Create;
   } else if constexpr (std::is_same_v<T, IRecorder>) {
      createFunctionMap["XmlRecorder"] = &XmlRecorder::Create;
      createFunctionMap["XmlGrowthRecorder"] = &XmlGrowthRecorder::Create;
      createFunctionMap["XmlSTDPRecorder"] = &XmlSTDPRecorder::Create;
      createFunctionMap["Xml911Recorder"] = &Xml911Recorder::Create;
#if defined(HDF5)
      createFunctionMap["Hdf5Recorder"] = &Hdf5Recorder::Create;
      createFunctionMap["Hdf5GrowthRecorder"] = &Hdf5GrowthRecorder::Create;
#endif
   } else if constexpr (std::is_same_v<T, MTRand>) {
      createFunctionMap["MTRand"] = &MTRand::Create;
      createFunctionMap["Norm"] = &Norm::Create;
   }
   return createFunctionMap;
}

template std::map<std::string, Connections *(*)(void)>
   getCreateFunctionForType<Connections>(Connections *);

template std::map<std::string, AllEdges *(*)(void)> getCreateFunctionForType<AllEdges>(AllEdges *);

template std::map<std::string, Layout *(*)(void)> getCreateFunctionForType<Layout>(Layout *);

template std::map<std::string, AllVertices *(*)(void)>
   getCreateFunctionForType<AllVertices>(AllVertices *);

template std::map<std::string, IRecorder *(*)(void)>
   getCreateFunctionForType<IRecorder>(IRecorder *);

template std::map<std::string, MTRand *(*)(void)> getCreateFunctionForType<MTRand>(MTRand *);

// register Connections classes
// template <>
// std::map<std::string, Connections *(*)(void)> getCreateFunctionForType(Connections *placeholder)
// {
//    std::map<std::string, Connections *(*)(void)> createFunctionMap;

//    createFunctionMap["ConnStatic"] = &ConnStatic::Create;
//    createFunctionMap["ConnGrowth"] = &ConnGrowth::Create;
//    createFunctionMap["Connections911"] = &Connections911::Create;

//    return createFunctionMap;
// }

// // register Edges classes
// template <>
// std::map<std::string, AllEdges *(*)(void)> getCreateFunctionForType(AllEdges *placeholder)
// {
//    std::map<std::string, AllEdges *(*)(void)> createFunctionMap;

//    createFunctionMap["AllSpikingSynapses"] = &AllSpikingSynapses::Create;
//    createFunctionMap["AllSTDPSynapses"] = &AllSTDPSynapses::Create;
//    createFunctionMap["AllDSSynapses"] = &AllDSSynapses::Create;
//    createFunctionMap["AllDynamicSTDPSynapses"] = &AllDynamicSTDPSynapses::Create;
//    createFunctionMap["All911Edges"] = &All911Edges::Create;

//    return createFunctionMap;
// }

// // register Layout classes
// template <> std::map<std::string, Layout *(*)(void)> getCreateFunctionForType(Layout *placeholder)
// {
//    std::map<std::string, Layout *(*)(void)> createFunctionMap;

//    createFunctionMap["FixedLayout"] = &FixedLayout::Create;
//    createFunctionMap["DynamicLayout"] = &DynamicLayout::Create;
//    createFunctionMap["Layout911"] = &Layout911::Create;

//    return createFunctionMap;
// }

// // register Vertices classes
// template <>
// std::map<std::string, AllVertices *(*)(void)> getCreateFunctionForType(AllVertices *placeholder)
// {
//    std::map<std::string, AllVertices *(*)(void)> createFunctionMap;

//    createFunctionMap["AllLIFNeurons"] = &AllLIFNeurons::Create;
//    createFunctionMap["AllIZHNeurons"] = &AllIZHNeurons::Create;
//    createFunctionMap["All911Vertices"] = &All911Vertices::Create;

//    return createFunctionMap;
// }

// // register Recorder classes
// template <>
// std::map<std::string, IRecorder *(*)(void)> getCreateFunctionForType(IRecorder *placeholder)
// {
//    std::map<std::string, IRecorder *(*)(void)> createFunctionMap;

//    createFunctionMap["XmlRecorder"] = &XmlRecorder::Create;
//    createFunctionMap["XmlGrowthRecorder"] = &XmlGrowthRecorder::Create;
//    createFunctionMap["XmlSTDPRecorder"] = &XmlSTDPRecorder::Create;
//    createFunctionMap["Xml911Recorder"] = &Xml911Recorder::Create;

// #if defined(HDF5)
//    createFunctionMap["Hdf5Recorder"] = &Hdf5Recorder::Create;
//    createFunctionMap["Hdf5GrowthRecorder"] = &Hdf5GrowthRecorder::Create;
// #endif

//    return createFunctionMap;
// }

// // register MTRand classes
// template <> std::map<std::string, MTRand *(*)(void)> getCreateFunctionForType(MTRand *placeholder)
// {
//    std::map<std::string, MTRand *(*)(void)> createFunctionMap;

//    createFunctionMap["MTRand"] = &MTRand::Create;
//    createFunctionMap["Norm"] = &Norm::Create;

//    return createFunctionMap;
// }
