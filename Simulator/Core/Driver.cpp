/**
 * @file Driver.cpp
 * 
 * @ingroup Simulator/Core
 *
 * @brief Orchestrates most functionality in the simulation.
 * 
 *  The driver performs the following steps:
 *  1) Instantiates Simulator object
 *  2) Parses command line to get configuration file and additional information if provided
 *  3) Loads global Simulator parameters from configuration file
 *  4) Instantiates all simulator objects (Layout, Connections, Synapases, Vertices)
 *  5) Reads simulator objects' parameters from configuration file
 *  6) Simulation setup (Deseralization, Initailizing values, etc.)
 *  7) Run Simulation
 *  8) Simulation shutdown (Save results, serialize)
 *
 */

#include <fstream>

#include "Global.h"
#include "../ThirdParty/paramcontainer/ParamContainer.h"
#include "log4cplus/logger.h"
#include "log4cplus/configurator.h"
#include "log4cplus/loggingmacros.h"

#include "AllEdges.h"
#include "CPUModel.h"
#include "Inputs/FSInput.h"
#include "IRecorder.h"
#include "Model.h"
#include "OperationManager.h"
#include "ParameterManager.h"
#include "Simulator.h"

#include <string>

// Uncomment to use visual leak detector (Visual Studios Plugin)
// #include <vld.h>

// Cereal
#include <cereal/archives/xml.hpp>
#include <cereal/archives/binary.hpp>
// TODO: fix this stuff
#include "ConnGrowth.h" // hacked in. that's why its here.
#include "ConnStatic.h" // hacked in. that's why its here.

// build/config.h contains the git commit id
#include "config.h"

#if defined(USE_GPU)
#include "GPUModel.h"
#elif defined(USE_OMP)
// #include "MultiThreadedSim.h"
#else

#endif

using namespace std;

// forward declarations
bool parseCommandLine(int argc, char *argv[]);
bool deserializeSynapses();
void serializeSynapses();

///  Main for Simulator. Handles command line arguments and loads parameters
///  from parameter file. All initial loading before running simulator in Network
///  is here.
///
///  @param  argc    argument count.
///  @param  argv    arguments.
///  @return -1 if error, else if success.
int main(int argc, char *argv[]) {

   // Clear logging files at the start of each simulation
   fstream("../Output/Debug/logging.txt", ios::out | ios::trunc);
   fstream("../Output/Debug/vertices.txt", ios::out | ios::trunc);
   fstream("../Output/Debug/edges.txt", ios::out | ios::trunc);

   // Initialize log4cplus and set properties based on configure file
   ::log4cplus::initialize();
   ::log4cplus::PropertyConfigurator::doConfigure("../RuntimeFiles/log4cplus_configure.ini");

   // Get the instance of the console logger and print status
   log4cplus::Logger consoleLogger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("console"));
   
   LOG4CPLUS_TRACE(consoleLogger, "Instantiating Simulator");
   Simulator &simulator = Simulator::getInstance();

   // Handles parsing of the command line.
   LOG4CPLUS_TRACE(consoleLogger, "Parsing command line");
   if (!parseCommandLine(argc, argv)) {
      LOG4CPLUS_FATAL(consoleLogger, "ERROR: failed during command line parse");
      return -1;
   }

   // Loads the configuration file into the Parameter Manager.
   if (!ParameterManager::getInstance().loadParameterFile(simulator.getConfigFileName())) {
      LOG4CPLUS_FATAL(consoleLogger, "ERROR: failed to load config file: " << simulator.getConfigFileName());
      return -1;
   }

   // Read in simulator specific parameters from configuration file.
   LOG4CPLUS_TRACE(consoleLogger, "Loading Simulator parameters");
   simulator.loadParameters();

   // Instantiate simulator objects.
   LOG4CPLUS_TRACE(consoleLogger, "Instantiating Simulator objects specified in configuration file");
   if (!simulator.instantiateSimulatorObjects()) {
      LOG4CPLUS_FATAL(consoleLogger, "ERROR: Unable to instantiate all simulator classes, check configuration file: "
                                     + simulator.getConfigFileName()
                                     + " for incorrectly declared class types.");
      return -1;
   }

   // Invoke instantiated simulator objects to load parameters from the configuration file
   LOG4CPLUS_TRACE(consoleLogger, "Loading parameters from configuration file");
   OperationManager::getInstance().executeOperation(Operations::loadParameters);

   time_t start_time, end_time;
   time(&start_time);

   // Setup simulation (calls model->setupSim)
   LOG4CPLUS_TRACE(consoleLogger, "Performing Simulator setup");
   simulator.setup();

   // Invoke instantiated simulator objects to print parameters, used for testing purposes only.
   OperationManager::getInstance().executeOperation(Operations::printParameters);

   // Deserializes internal state from a prior run of the simulation
   if (!simulator.getDeserializationFileName().empty()) {
      LOG4CPLUS_TRACE(consoleLogger, "Deserializing state from file.");

      // Deserialization
      if (!deserializeSynapses()) {
         LOG4CPLUS_FATAL(consoleLogger, "Failed while deserializing objects");
         return -1;
      }
   }

   // Run simulation
   LOG4CPLUS_TRACE(consoleLogger, "Starting Simulation");
   simulator.simulate();

   // INPUT OBJECTS ARENT IN PROJECT YET
   // Terminate the stimulus input
//   if (pInput != nullptr)
//   {
//      simInfo->pInput->term();
//      delete simInfo->pInput;
//   }

   // todo: before this, do copy from gpu.
   // Writes simulation results to an output destination
   LOG4CPLUS_TRACE(consoleLogger, "Simulation ended, saving results");
   simulator.saveResults();

   // todo: going to be moved with the "hack"
   // Serializes internal state for the current simulation
   if (!simulator.getSerializationFileName().empty()) {
      LOG4CPLUS_TRACE(consoleLogger, "Serializing current state");
      serializeSynapses();
   }

   // Tell simulation to clean-up and run any post-simulation logic.
   LOG4CPLUS_TRACE(consoleLogger, "Simulation finished");
   simulator.finish();

   // terminates the simulation recorder
   if (simulator.getModel()->getRecorder() != nullptr) {
      simulator.getModel()->getRecorder()->term();
   }

   delete noiseRNG;

   time(&end_time);
   double timeElapsed = difftime(end_time, start_time);
   double ssps = simulator.getEpochDuration() * simulator.getNumEpochs() / timeElapsed;
   cout << "time simulated: " << simulator.getEpochDuration() * simulator.getNumEpochs() << endl;
   cout << "time elapsed: " << timeElapsed << endl;
   cout << "ssps (simulation seconds / real time seconds): " << ssps << endl;
   return 0;
}

///  Handles parsing of the command line
///
///  @param  argc      argument count.
///  @param  argv      arguments.
///  @returns    true if successful, false otherwise.
bool parseCommandLine(int argc, char *argv[]) {
   ParamContainer cl; // todo: note as third party class.
   cl.initOptions(false);  // don't allow unknown parameters
   cl.setHelpString(string(
         "The UW Bothell graph-based simulation environment, for high-performance neural network and other graph-based problems.\nUsage: ") +
                    argv[0] + " ");

   // Set up the comment line parser.
   if ((cl.addParam("resultfile", 'o', ParamContainer::filename, "simulation results filepath (deprecated)") !=
        ParamContainer::errOk)
       || (cl.addParam("configfile", 'c', ParamContainer::filename,
                       "parameter configuration filepath") != ParamContainer::errOk)
       #if defined(USE_GPU)
       || (cl.addParam("device", 'd', ParamContainer::regular, "CUDA device id") != ParamContainer::errOk)
       #endif  // USE_GPU
       ||
       (cl.addParam("stimulusfile", 's', ParamContainer::filename, "stimulus input filepath") != ParamContainer::errOk)
       || (cl.addParam("deserializefile", 'r', ParamContainer::filename,
                       "simulation deserialization filepath (enables deserialization)") != ParamContainer::errOk)
       || (cl.addParam("serializefile", 'w', ParamContainer::filename,
                       "simulation serialization filepath (enables serialization)") != ParamContainer::errOk)
       || (cl.addParam("version", 'v', ParamContainer::novalue,
                       "output current git commit ID and exit") != ParamContainer::errOk)) {

      cerr << "Internal error creating command line parser" << endl;
      return false;
   }

   // Parse the command line
   if (cl.parseCommandLine(argc, argv) != ParamContainer::errOk) {
      cl.dumpHelp(stderr, true, 78);
      return false;
   }

   if (cl["version"].compare("") != 0) {
      cout << "Git commit ID: " << GIT_COMMIT_ID << endl;
      exit(0);
   }

   // Get the command line values
   Simulator::getInstance().setResultFileName(cl["resultfile"]);
   Simulator::getInstance().setConfigFileName(cl["configfile"]);
   Simulator::getInstance().setDeserializationFileName(cl["deserializefile"]);
   Simulator::getInstance().setSerializationFileName(cl["serializefile"]);
   Simulator::getInstance().setStimulusFileName(cl["stimulusfile"]);

#if defined(USE_GPU)
   if (EOF == sscanf(cl["device"].c_str(), "%d", &g_deviceId)) {
       g_deviceId = 0;
   }
#endif  // USE_GPU
   return true;
}

///  Deserializes synapse weights, source vertices, destination vertices,
///  maxEdgesPerVertex, totalVertices, and
///  if running a connGrowth model and radii is in serialization file, deserializes radii as well
///
///  @returns    true if successful, false otherwise.
bool deserializeSynapses() {
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
   }
   catch (cereal::Exception e) {
      cerr << "Failed deserializing synapse weights, source vertices, and/or destination vertices." << endl;
      return false;
   }

   // Creates synapses from weight
   connections->createSynapsesFromWeights(simulator.getTotalVertices(),
                                          layout.get(),
                                          (*layout->getVertices()),
                                          (*connections->getEdges()));


#if defined(USE_GPU)
   // Copies CPU Synapse data to GPU after deserialization, if we're doing
   // a GPU-based simulation.
   simulator.copyCPUSynapseToGPU();
#endif // USE_GPU

   // Creates synapse index map (includes copy CPU index map to GPU)
   connections->createEdgeIndexMap();

#if defined(USE_GPU)
   GPUModel *gpuModel = static_cast<GPUModel *>(simulator.getModel().get());
   gpuModel->copySynapseIndexMapHostToDevice(*(connections->getEdgeIndexMap().get()), simulator.getTotalVertices());
#endif // USE_GPU

   // Deserializes radii (only when running a connGrowth model and radii is in serialization file)
   if (dynamic_cast<ConnGrowth *>(connections.get()) != nullptr) {
      // Uses "try catch" to catch any cereal exception
      try {
         archive(*(dynamic_cast<ConnGrowth *>(connections.get())));
      }
      catch (cereal::Exception e) {
         cerr << "Failed deserializing radii." << endl;
         return false;
      }
   }
   return true;
}

void serializeSynapses() {
   Simulator &simulator = Simulator::getInstance();

   // We can serialize to a variety of archive file formats. Below, comment out
   // all but the two lines that correspond to the desired format.
   ofstream memory_out(simulator.getSerializationFileName().c_str());
   cereal::XMLOutputArchive archive(memory_out);
   //ofstream memory_out (simInfo->memOutputFileName.c_str(), std::ios::binary);
   //cereal::BinaryOutputArchive archive(memory_out);
   
#if defined(USE_GPU)
   // Copies GPU Synapse props data to CPU for serialization
   simulator.copyGPUSynapseToCPU();
#endif // USE_GPU
   
    shared_ptr<Model> model = simulator.getModel();

   // Serializes synapse weights along with each synapse's source vertex and destination vertex
   archive(*(dynamic_cast<AllEdges *>(model->getConnections()->getEdges().get())));

   // Serializes radii (only if it is a connGrowth model)
   if (dynamic_cast<ConnGrowth *>(model->getConnections().get()) != nullptr) {
      archive(*(dynamic_cast<ConnGrowth *>(model->getConnections().get())));
   }
}
