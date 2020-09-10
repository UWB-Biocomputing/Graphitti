/**
 * @file Driver
 *
 *  The driver performs the following steps:
 *  1) Instantiates Simulator object
 *  2) Parses command line to get configuration file and additional information if provided
 *  3) Loads global Simulator parameters from configuration file
 *  4) Instantiates all simulator objects (Layout, Connections, Synapases, Neurons)
 *  5) Reads simulator objects' parameters from configuration file
 *  6) Simulation setup (Deseralization, Initailizing values, etc.)
 *  7) Run Simulation
 *  8) Simulation shutdown (Save results, serialize)
 *
 */

#include <fstream>

#include "Global.h"
#include "ThirdParty/paramcontainer/ParamContainer.h"
#include "log4cplus/logger.h"
#include "log4cplus/configurator.h"
#include "log4cplus/loggingmacros.h"

#include "AllSynapses.h"
#include "CPUSpikingModel.h"
#include "Inputs/FSInput.h"
#include "IRecorder.h"
#include "Model.h"
#include "OperationManager.h"
#include "ParameterManager.h"
#include "Simulator.h"


// Uncomment to use visual leak detector (Visual Studios Plugin)
// #include <vld.h>

//! Cereal
#include <cereal/archives/xml.hpp>
#include <cereal/archives/binary.hpp>
#include "ConnGrowth.h" // hacked in. that's why its here.

#if defined(USE_GPU)
#include "GPUSpikingModel.h"
#elif defined(USE_OMP)
// #include "MultiThreadedSim.h"
#else

#endif

using namespace std;

// forward declarations
bool parseCommandLine(int argc, char *argv[]);
bool deserializeSynapses();
void serializeSynapses();

/*
 *  Main for Simulator. Handles command line arguments and loads parameters
 *  from parameter file. All initial loading before running simulator in Network
 *  is here.
 *
 *  @param  argc    argument count.
 *  @param  argv    arguments.
 *  @return -1 if error, else if success.
 */
int main(int argc, char *argv[]) {
   // Clear logging file at the start of each simulation
   fstream("Output/Debug/logging.txt", ios::out | ios::trunc);

   // Initialize log4cplus and set properties based on configure file
   ::log4cplus::initialize();
   ::log4cplus::PropertyConfigurator::doConfigure("RuntimeFiles/log4cplus_configure.ini");

   // Get the instance of the main logger and print status
   log4cplus::Logger logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("main"));
   LOG4CPLUS_TRACE(logger, "Initiating Simulator");

   Simulator &simulator = Simulator::getInstance();

   // Handles parsing of the command line.
   LOG4CPLUS_TRACE(logger, "Parsing command line");
   if (!parseCommandLine(argc, argv)) {
      LOG4CPLUS_FATAL(logger, "ERROR: failed during command line parse");
      return -1;
   }

   // Loads the configuration file into the Parameter Manager.
   if (!ParameterManager::getInstance().loadParameterFile(simulator.getConfigFileName())) {
      LOG4CPLUS_FATAL(logger, "ERROR: failed to load config file: " << simulator.getConfigFileName());
      return -1;
   }

   // Read in simulator specific parameters from configuration file.
   LOG4CPLUS_TRACE(logger, "Loading Simulator parameters");
   simulator.loadParameters();

   // Instantiate simulator objects.
   LOG4CPLUS_TRACE(logger, "Insantiating Simulator objects specified in configuration file");
   if (!simulator.instantiateSimulatorObjects()) {
      LOG4CPLUS_FATAL(logger, "ERROR: Unable to instantiate all simulator classes, check configuration file: "
                              + simulator.getConfigFileName()
                              + " for incorrectly declared class types.");
      return -1;
   }

<<<<<<< HEAD
   // Have instantiated simulator objects load parameters from the configuration file
   OperationManager::getInstance().executeOperation(Operations::op::loadParameters);

   cout << "Printing Layout Params" << endl;
   simulator.getModel()->getLayout()->printParameters();
   cout << endl;

   cout << "Printing Neuron Params" << endl;
   simulator.getModel()->getLayout()->getNeurons()->printParameters();
   cout << endl;

   cout << "Printing Connections Params" << endl;
   simulator.getModel()->getConnections()->printParameters();

    cout << "Printing Synapse Params" << endl;
   simulator.getModel()->getConnections()->getSynapses()->printParameters();


//   time_t start_time, end_time;
//   time(&start_time);
//
//   // in chain of responsibility. still going to exist!
//   // setup simulation
//   DEBUG(cerr << "Setup simulation." << endl;)
//   simulator.setup();
//
//   // Deserializes internal state from a prior run of the simulation
//   if (!simInfo->memInputFileName.empty()) {
//      DEBUG(cerr << "Deserializing state from file." << endl;)
//
//      DEBUG(
//      // Prints out internal state information before deserialization
//            cout << "------------------------------Before Deserialization:------------------------------" << endl;
//            printKeyStateInfo();
//      )
//
//      // Deserialization
//      if(!deserializeSynapseInfo()) {
//         cerr << "! ERROR: failed while deserializing objects" << endl;
//         return -1;
//      }
//
//      DEBUG(
//      // Prints out internal state information after deserialization
//            cout << "------------------------------After Deserialization:------------------------------" << endl;
//            printKeyStateInfo(simInfo);
//      )
//   }
//
//   // Run simulation
//   simulator->simulate();
//
//   // todo:put someplace else, like chain of responsibility. doesnt have to happen here.
//   // Terminate the stimulus input
//   if (simInfo->pInput != NULL)
=======
   // Invoke instantiated simulator objects to load parameters from the configuration file
   LOG4CPLUS_TRACE(logger, "Loading parameters from configuration file");
   OperationManager::getInstance().executeOperation(Operations::loadParameters);

   time_t start_time, end_time;
   time(&start_time);

   // in chain of responsibility. still going to exist!
   // setup simulation
   LOG4CPLUS_TRACE(logger, "Performing Simulator setup");
   simulator.setup();

   // Invoke instantiated simulator objects to print parameters, used for testing purposes only.
   OperationManager::getInstance().executeOperation(Operations::printParameters);

   // Deserializes internal state from a prior run of the simulation
   if (!simulator.getSerializationFileName().empty()) {
      LOG4CPLUS_TRACE(logger, "Deserializing state from file.");

      // Deserialization
      if (!deserializeSynapses()) {
         LOG4CPLUS_FATAL(logger, "Failed while deserializing objects");
         return -1;
      }
   }

   // Run simulation
   LOG4CPLUS_TRACE(logger, "Starting Simulation");
   simulator.simulate();

   // INPUT OBJECTS ARENT IN PROJECT YET
   // Terminate the stimulus input
//   if (pInput != NULL)
>>>>>>> e311573ca2647649ee14566d3a3788b42aa1b6d0
//   {
//      simInfo->pInput->term();
//      delete simInfo->pInput;
//   }

   // todo: before this, do copy from gpu.
   // Writes simulation results to an output destination
   LOG4CPLUS_TRACE(logger, "Simulation ended, saving results");
   simulator.saveData();

   // todo: going to be moved with the "hack"
   // Serializes internal state for the current simulation
   if (!simulator.getSerializationFileName().empty()) {
      LOG4CPLUS_TRACE(logger, "Serializing current state");
      serializeSynapses();
   }

   // Tell simulation to clean-up and run any post-simulation logic.
   LOG4CPLUS_TRACE(logger, "Simulation finished");
   simulator.finish();

   // terminates the simulation recorder
   if (simulator.getModel()->getRecorder() != nullptr) {
      simulator.getModel()->getRecorder()->term();
   }

   for (unsigned int i = 0; i < rgNormrnd.size(); ++i) {
      delete rgNormrnd[i];
   }

   rgNormrnd.clear();

   time(&end_time);
   double time_elapsed = difftime(end_time, start_time);
   double ssps = simulator.getEpochDuration() * simulator.getNumEpochs() / time_elapsed;
   cout << "time simulated: " << simulator.getEpochDuration() * simulator.getNumEpochs() << endl;
   cout << "time elapsed: " << time_elapsed << endl;
   cout << "ssps (simulation seconds / real time seconds): " << ssps << endl;
   return 0;
}

/*
 *  Handles parsing of the command line
 *
 *  @param  argc      argument count.
 *  @param  argv      arguments.
 *  @returns    true if successful, false otherwise.
 */
bool parseCommandLine(int argc, char *argv[]) {
   ParamContainer cl; // todo: note as third party class.
   cl.initOptions(false);  // don't allow unknown parameters
   cl.setHelpString(string(
         "The UW Bothell graph-based simulation environment, for high-performance neural network and other graph-based problems.\nUsage: ") +
                    argv[0] + " ");

   // Set up the comment line parser.
   if ((cl.addParam("resultfile", 'o', ParamContainer::filename, "simulation results filepath (deprecated)") !=
        ParamContainer::errOk)
       || (cl.addParam("configfile", 'c', ParamContainer::filename | ParamContainer::required,
                       "parameter configuration filepath") != ParamContainer::errOk)
       #if defined(USE_GPU)
       || (cl.addParam("device", 'd', ParamContainer::regular, "CUDA device id") != ParamContainer::errOk)
       #endif  // USE_GPU
       ||
       (cl.addParam("stimulusfile", 's', ParamContainer::filename, "stimulus input filepath") != ParamContainer::errOk)
       || (cl.addParam("deserializefile", 'r', ParamContainer::filename,
                       "simulation deserialization filepath (enables deserialization)") != ParamContainer::errOk)
       || (cl.addParam("serializefile", 'w', ParamContainer::filename,
                       "simulation serialization filepath (enables serialization)") != ParamContainer::errOk)) {
      cerr << "Internal error creating command line parser" << endl;
      return false;
   }

   // Parse the command line
   if (cl.parseCommandLine(argc, argv) != ParamContainer::errOk) {
      cl.dumpHelp(stderr, true, 78);
      return false;
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

/*
 *  Deserializes synapse weights, source neurons, destination neurons,
 *  maxSynapsesPerNeuron, totalNeurons, and
 *  if running a connGrowth model and radii is in serialization file, deserializes radii as well
 *
 *  @returns    true if successful, false otherwise.
 */
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

   // We can deserialize from a variety of archive file formats. Below, commentp
   // out all but the line that corresponds to the desired format.
   cereal::XMLInputArchive archive(memory_in);
   //cereal::BinaryInputArchive archive(memory_in);

   shared_ptr<Connections> connections = simulator.getModel()->getConnections();
   shared_ptr<Layout> layout = simulator.getModel()->getLayout();

   if (!layout || !connections) {
      cerr << "Either connections or layout is not instantiated," << endl;
   }


   // Deserializes synapse weights along with each synapse's source neuron and destination neuron
   // Uses "try catch" to catch any cereal exception
   try {
      archive(*(dynamic_cast<AllSynapses *>(connections->getSynapses().get())));
   }
   catch (cereal::Exception e) {
      cerr << "Failed deserializing synapse weights, source neurons, and/or destination neurons." << endl;
      return false;
   }

   // Creates synapses from weight
   connections->createSynapsesFromWeights(simulator.getTotalNeurons(), layout.get(), (*layout->getNeurons()),
                                          (*connections->getSynapses()));

#if defined(USE_GPU)
   // Copies CPU Synapse data to GPU after deserialization, if we're doing
    // a GPU-based simulation.
    simulator.copyCPUSynapseToGPU();
#endif // USE_GPU

   // Creates synapse index map (includes copy CPU index map to GPU)
   connections->createSynapseIndexMap();

#if defined(USE_GPU)
   dynamic_cast<GPUSpikingModel *>(simInfo->model)->copySynapseIndexMapHostToDevice(*(dynamic_cast<GPUSpikingModel *>(simInfo->model)->m_synapseIndexMap), simInfo->totalNeurons);
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
    simulator->copyGPUSynapseToCPU();
#endif // USE_GPU
    shared_ptr<Model> model = simulator.getModel();

   // Serializes synapse weights along with each synapse's source neuron and destination neuron
   archive(*(dynamic_cast<AllSynapses *>(model->getConnections()->getSynapses().get())));

   // Serializes radii (only if it is a connGrowth model)
   if (dynamic_cast<ConnGrowth *>(model->getConnections().get()) != nullptr) {
      archive(*(dynamic_cast<ConnGrowth *>(model->getConnections().get())));
   }
}
