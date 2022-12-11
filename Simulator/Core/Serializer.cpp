/**
 * @file Serializer.cpp
 *
 * @ingroup Simulator/Core
 * 
 * @brief Handles implementation details of serialization and deserialization of synapses.
 * 
 *  Serialize and Deserialize synapse weights, source vertices, destination vertices,
 *  maxEdgesPerVertex, totalVertices and radii.
 */

#include "Serializer.h"
#include "ConnGrowth.h"
#include "Connections.h"
#include "GPUModel.h"
#include "Simulator.h"
#include <fstream>

// Displays Graphitti as top most element instead of the default Cereal
// CEREAL_XML_STRING_VALUE should be placed before defining cereal archives library
#define CEREAL_XML_STRING_VALUE "Graphitti"
#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>

///  Deserializes synapse weights, source vertices, destination vertices,
///  maxEdgesPerVertex, totalVertices.
///  if running a connGrowth model and radii is in serialization file, deserializes radii as well
///
///  @returns    true if successful, false otherwise.
bool Serializer::deserializeSynapses()
{
   Simulator &simulator = Simulator::getInstance();

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

   shared_ptr<Connections> connections = simulator.getModel()->getConnections();
   shared_ptr<Layout> layout = simulator.getModel()->getLayout();

   if (!layout || !connections) {
      cerr << "Either connections or layout is not instantiated," << endl;
   }

   // Deserializes synapse weights along with each synapse's source vertex and destination vertex
   // Uses "try catch" to catch any cereal exception
   try {
      archive(*(dynamic_cast<AllEdges *>(connections->getEdges().get())));
   } catch (cereal::Exception e) {
      cerr << "Failed deserializing synapse weights, source vertices, and/or destination vertices."
           << endl;
      return false;
   }

   // Creates synapses from weight
   //    connections->createSynapsesFromWeights(simulator.getTotalVertices(), layout.get(),
   //                                           (*layout->getVertices()), (*connections->getEdges()));

   //Creates synapses from weight
   connections->createSynapsesFromWeights();


#if defined(USE_GPU)
   // Copies CPU Synapse data to GPU after deserialization, if we're doing
   // a GPU-based simulation.
   simulator.copyCPUSynapseToGPU();
#endif   // USE_GPU

   // Creates synapse index map (includes copy CPU index map to GPU)
   connections->createEdgeIndexMap();

#if defined(USE_GPU)
   GPUModel *gpuModel = static_cast<GPUModel *>(simulator.getModel().get());
   gpuModel->copySynapseIndexMapHostToDevice(*(connections->getEdgeIndexMap().get()),
                                             simulator.getTotalVertices());
#endif   // USE_GPU

   // Deserializes radii (only when running a connGrowth model and radii is in serialization file)
   if (dynamic_cast<ConnGrowth *>(connections.get()) != nullptr) {
      // Uses "try catch" to catch any cereal exception
      try {
         archive(*(dynamic_cast<ConnGrowth *>(connections.get())));
      } catch (cereal::Exception e) {
         cerr << "Failed deserializing radii." << endl;
         return false;
      }
   }
   return true;
}

///  Serializes synapse weights, source vertices, destination vertices,
///  maxEdgesPerVertex, totalVertices.
///  if running a connGrowth model serializes radii as well.
void Serializer::serializeSynapses()
{
   Simulator &simulator = Simulator::getInstance();

   // We can serialize to a variety of archive file formats. Below, comment out
   // all but the two lines that correspond to the desired format.
   ofstream memory_out(simulator.getSerializationFileName().c_str());
   cout << "PLease find the serialized file in " << simulator.getSerializationFileName().c_str();

   // Options parameter are optional which sets
   // 1. Sets the Preceision of floating point number to 30
   // 2. Keeps indedntaion in XML file,
   // 3. Displays output type of element values in XML eg. float
   // 4. Displays if the size of a node in XML is dynamic or not.
   cereal::XMLOutputArchive archive(memory_out, cereal::XMLOutputArchive::Options()
                                                   .precision(30)
                                                   .indent(true)
                                                   .outputType(false)
                                                   .sizeAttributes(true));
   //ofstream memory_out (simInfo->memOutputFileName.c_str(), std::ios::binary);
   //cereal::BinaryOutputArchive archive(memory_out);

#if defined(USE_GPU)
   // Copies GPU Synapse props data to CPU for serialization
   simulator.copyGPUSynapseToCPU();
#endif   // USE_GPU

   shared_ptr<Model> model = simulator.getModel();

   // Serializes synapse weights along with each synapse's source vertex and destination vertex
   archive(cereal::make_nvp(
      "AllEdges", *(dynamic_cast<AllEdges *>(model->getConnections()->getEdges().get()))));

   // Serializes radii (only if it is a connGrowth model)
   if (dynamic_cast<ConnGrowth *>(model->getConnections().get()) != nullptr) {
      archive(cereal::make_nvp("Connections",
                               *(dynamic_cast<ConnGrowth *>(model->getConnections().get()))));
   }
}
