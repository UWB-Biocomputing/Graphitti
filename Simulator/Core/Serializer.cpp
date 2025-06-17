/**
 * @file Serializer.cpp
 *
 * @ingroup Simulator/Core
 * 
 * @brief Provides serialization and deserialization functionality using the Cereal library.
 * 
 * This class handles the serialization and deserialization of all member variables 
 * in the Connections, Layout, Edges, Vertices, and associated helper classes such as
 * EdgeIndexMap, Model, RecordableBase, RecordableVector, Matrix, RNG and EventBuffer.
 * Note that Recorder class is not serialized or deserialized.
 * 
 * The serialization and deserialization process typically begins with the Model class,
 * which internally calls the serialization of the Connections and Layout classes.
 * Connections, in turn, handle the serialization of Edges, while Layout handles 
 * the serialization of Vertices. This ensures a comprehensive serialization of 
 * the entire simulation structure.
 * 
 * @note As of September 2024, serialization support is currently available
 * for CPU-based Neuron simulations. While GPU-based Neuron serialization is functional, 
 * the output result files differ, and this is being addressed in [Issue #701].
 * Serialization support for NG911 will be extended in the future [Issue #700].
 *  
 */

#include "Serializer.h"
#include "ConnGrowth.h"
#include "GPUModel.h"
#include <fstream>

// About CEREAL_XML_STRING_VALUE
// 1. Displays Graphitti as top most element instead of the default Cereal in the serialized xml file.
// 2. It should be placed before defining cereal archives library
#define CEREAL_XML_STRING_VALUE "Graphitti"
#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>

/// Deserializes all member variables of the
/// Connections, Layout, Edges, Vertices, and associated helper classes.
///
///  @returns    true if successful, false otherwise.
bool Serializer::deserialize()
{
   Simulator &simulator = Simulator::getInstance();

   // We can deserialize from a variety of archive file formats. Below, comment
   // out all but the line that is compatible with the desired format.
   ifstream memory_in(simulator.getDeserializationFileName().c_str());
   //ifstream memory_in (simInfo->memInputFileName.c_str(), std::ios::binary);
   log4cplus::Logger consoleLogger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("console"));
   
   // Checks to see if serialization file exists
   if (!memory_in) {
      LOG4CPLUS_FATAL(consoleLogger,"The serialization file doesn't exist");
      return false;
   }

   // We can deserialize from a variety of archive file formats. Below, comment
   // out all but the line that corresponds to the desired format.
   cereal::XMLInputArchive archive(memory_in);
   //cereal::BinaryInputArchive archive(memory_in);

   if (!processArchive(archive, simulator)) {
      LOG4CPLUS_FATAL(consoleLogger, "Failed to deserialize");
      return false;
   }


#if defined(USE_GPU)
   GPUModel &gpuModel = static_cast<GPUModel &>(simulator.getModel());
   gpuModel.copyEdgeIndexMapHostToDevice(simulator.getModel().getConnections().getEdgeIndexMap(),
                                         simulator.getTotalVertices());
#endif   // USE_GPU

   return true;
}

/// Serializes all member variables of the
/// Connections, Layout, Edges, Vertices, and associated helper classes.
void Serializer::serialize()
{
   Simulator &simulator = Simulator::getInstance();

   log4cplus::Logger consoleLogger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("console"));

   // We can serialize to a variety of archive file formats. Below, comment out
   // all but the two lines that correspond to the desired format.
   ofstream memory_out(simulator.getSerializationFileName().c_str());
   string message = "Please find the serialized file in " + simulator.getSerializationFileName();
   LOG4CPLUS_TRACE(consoleLogger, message);
   cereal::XMLOutputArchive archive(memory_out);
   //ofstream memory_out (simInfo->memOutputFileName.c_str(), std::ios::binary);
   //cereal::BinaryOutputArchive archive(memory_out);

   if (!processArchive(archive, simulator)) {
      LOG4CPLUS_ERROR(consoleLogger, "Failed to serialize");
   }
}

template <typename Archive> bool Serializer::processArchive(Archive &archive, Simulator &simulator)
{
   try {
      // Starts the serialization/deserialization process from the Model class.
      // Note that the Model object gets sliced, and only
      // the `serialize` function of the base Model class is called.
      archive(simulator.getModel());
      // Serialize/Deserialize required global variables
      archive(initRNG, noiseRNG, g_simulationStep);
   } catch (cereal::Exception e) {
      log4cplus::Logger consoleLogger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("console"));
      LOG4CPLUS_ERROR(consoleLogger, e.what());
      return false;
   }
   return true;
}
