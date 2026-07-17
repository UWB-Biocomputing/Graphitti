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
#include "AllEdges.h"
#include "ConnGrowth.h"
#include "Connections.h"
#include "Factory.h"
#include "GPUModel.h"
#include "Model.h"
#include "OperationManager.h"
#include "ParameterManager.h"
#include <fstream>

// About CEREAL_XML_STRING_VALUE
// 1. Displays Graphitti as top most element instead of the default Cereal in the serialized xml file.
// 2. It should be placed before defining cereal archives library
#define CEREAL_XML_STRING_VALUE "Graphitti"
#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>

namespace {

   /// Imports the grown network topology from a deserialized ConnGrowth checkpoint into a
   /// freshly constructed Connections object of the type requested by the current run's
   /// configuration file (for example ConnStatic with AllSTDPSynapses).
   ///
   /// This enables the output network of a growth simulation to be used as the starting
   /// point ("input") for a subsequent STDP simulation: only the edge source, destination,
   /// weight, and type are carried over. The restored vertices/layout and global simulation
   /// state (RNG, simulation step) are left untouched.
   ///
   /// @param connectionClassName  Connections class named in the current configuration file.
   void importGrowthTopology(const string &connectionClassName)
   {
      Simulator &simulator = Simulator::getInstance();
      Model &model = simulator.getModel();

      // Edges grown during the checkpointed growth simulation (still owned by the model).
      AllEdges &grownEdges = model.getConnections().getEdges();

      // Build the Connections/Edges objects requested by the current configuration file.
      unique_ptr<Connections> importedConnections
         = Factory<Connections>::getInstance().createType(connectionClassName);
      if (importedConnections == nullptr) {
         throw runtime_error("Deserialization topology import: unknown Connections class '"
                             + connectionClassName + "'");
      }

      AllEdges &importedEdges = importedConnections->getEdges();
      importedEdges.setupEdges();
      // Populate per-edge parameters (e.g. STDP constants) from the configuration file so that
      // addEdge()/createEdge() initialize the new edges with the correct values.
      importedEdges.loadParameters();

      BGFLOAT deltaT = simulator.getDeltaT();
      BGSIZE importedCount = 0;
      for (BGSIZE iEdg = 0; iEdg < grownEdges.inUse_.size(); iEdg++) {
         if (grownEdges.inUse_[iEdg] == 0) {
            continue;
         }
         int srcVertex = grownEdges.sourceVertexIndex_[iEdg];
         int destVertex = grownEdges.destVertexIndex_[iEdg];
         edgeType type = grownEdges.type_[iEdg];
         BGSIZE newEdg = importedEdges.addEdge(type, srcVertex, destVertex, deltaT);
         importedEdges.W_[newEdg] = grownEdges.W_[iEdg];
         ++importedCount;
      }

      // Install the new connection subgraph (destroys the checkpoint's ConnGrowth) and rebuild
      // its edge index map from the imported edges.
      model.setConnections(std::move(importedConnections));
      model.getConnections().createEdgeIndexMap();

      log4cplus::Logger consoleLogger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("console"));
      LOG4CPLUS_INFO(consoleLogger, "Imported " << importedCount << " grown edges into a "
                                                << connectionClassName
                                                << " network for the current simulation.");
   }

}   // namespace

/// Deserializes all member variables of the
/// Connections, Layout, Edges, Vertices, and associated helper classes.
///
///  @returns    true if successful, false otherwise.
bool Serializer::deserialize()
{
   Simulator &simulator = Simulator::getInstance();
   OperationManager &opsManager = OperationManager::getInstance();

   // We can deserialize from a variety of archive file formats. Below, comment
   // out all but the line that is compatible with the desired format.
   ifstream memory_in(simulator.getDeserializationFileName().c_str());
   //ifstream memory_in (simInfo->memInputFileName.c_str(), std::ios::binary);

   // Checks to see if serialization file exists
   if (!memory_in) {
      cerr << "The serialization file doesn't exist" << endl;
      return false;
   }

   // We can deserialize from a variety of archive file formats. Below, comment
   // out all but the line that corresponds to the desired format.
   cereal::XMLInputArchive archive(memory_in);
   //cereal::BinaryInputArchive archive(memory_in);

   if (!processArchive(archive, simulator)) {
      cerr << "Failed to deserialize" << endl;
      return false;
   }

   // If a growth checkpoint is being loaded into a non-growth (e.g. STDP) configuration,
   // carry over only the grown topology rather than resuming the growth model. This is what
   // enables using a growth simulation's output network as the input for an STDP simulation.
   string connectionClassName;
   ParameterManager::getInstance().getStringByXpath("//ConnectionsParams/@class",
                                                    connectionClassName);
   bool checkpointIsGrowth
      = dynamic_cast<ConnGrowth *>(&simulator.getModel().getConnections()) != nullptr;
   if (checkpointIsGrowth && connectionClassName != "ConnGrowth") {
      importGrowthTopology(connectionClassName);
   }

   // Deserialization rebuilds Connections/Layout subgraphs (and nested edges_/vertices_
   // unique_ptrs). Constructors register OperationManager callbacks via std::bind(this, ...),
   // but destroyed objects leave stale entries that segfault on the next executeOperation().
   // Clear the callback list and re-register from the live Simulator/Model objects only.
   opsManager.clearRegisteredOperations();
   simulator.registerOperations();
   simulator.getModel().registerOperations();

#if defined(USE_GPU)
   // setupSim() already allocated GPU memory for the pre-checkpoint state. Rebuild device
   // buffers so they match the deserialized host Connections/Layout subgraph.
   GPUModel &gpuModel = static_cast<GPUModel &>(simulator.getModel());
   gpuModel.reinitializeDeviceAfterDeserialize();
#endif   // USE_GPU

   return true;
}

/// Serializes all member variables of the
/// Connections, Layout, Edges, Vertices, and associated helper classes.
void Serializer::serialize()
{
   Simulator &simulator = Simulator::getInstance();

   // We can serialize to a variety of archive file formats. Below, comment out
   // all but the two lines that correspond to the desired format.
   ofstream memory_out(simulator.getSerializationFileName().c_str());
   cout << "Please find the serialized file in " << simulator.getSerializationFileName().c_str();

   cereal::XMLOutputArchive archive(memory_out);
   //ofstream memory_out (simInfo->memOutputFileName.c_str(), std::ios::binary);
   //cereal::BinaryOutputArchive archive(memory_out);

   if (!processArchive(archive, simulator)) {
      cerr << "Failed to serialize" << endl;
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
      cerr << e.what() << endl;
      return false;
   }
   return true;
}