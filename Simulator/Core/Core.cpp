/**
 * @file Core.cpp
 * 
 * @ingroup Simulator/Core
 *
 * @brief Orchestrates most functionality in the simulation.
 * 
 *  The runSimulation method performs the following steps:
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

#include "Core.h"
#include "../ThirdParty/paramcontainer/ParamContainer.h"
#include "ConnStatic.h"   //TODO: fix this stuff hacked in. that's why its here.
#include "GraphManager.h"
#include "OperationManager.h"
#include "ParameterManager.h"
#include "Serializer.h"
#include "config.h"   // build/config.h contains the git commit id

// Uncomment to use visual leak detector (Visual Studios Plugin)
// #include <vld.h>
#if defined(USE_GPU)
   #include "GPUModel.h"
#elif defined(USE_OMP)
// #include "MultiThreadedSim.h"
#else

#endif

using namespace std;

///  Handles parsing of the command line
///
///  @param  cmdLineArguments command line argument
///  @param executableName Name of the simultaor's executable file
///  @returns    true if successful, false otherwise.
bool Core::parseCommandLine(string executableName, string cmdLineArguments)
{
   ParamContainer cl;       // todo: note as third party class.
   cl.initOptions(false);   // don't allow unknown parameters
   cl.setHelpString(string(
      "The UW Bothell graph-based simulation environment, for high-performance neural network and other graph-based problems\n Usage: "
      + executableName + " "));

   // Set up the comment line parser.
   if ((cl.addParam("configfile", 'c', ParamContainer::filename, "parameter configuration filepath")
        != ParamContainer::errOk)
#if defined(USE_GPU)
       || (cl.addParam("device", 'g', ParamContainer::regular, "CUDA GPU device ID")
           != ParamContainer::errOk)
#endif   // USE_GPU
       || (cl.addParam("inputfile", 'i', ParamContainer::filename, "input file path")
           != ParamContainer::errOk)
       || (cl.addParam("deserializefile", 'd', ParamContainer::filename,
                       "simulation deserialization filepath (enables deserialization)")
           != ParamContainer::errOk)
       || (cl.addParam("serializefile", 's', ParamContainer::filename,
                       "simulation serialization filepath (enables serialization)")
           != ParamContainer::errOk)
       || (cl.addParam("version", 'v', ParamContainer::novalue,
                       "output current git commit ID and exit")
           != ParamContainer::errOk)) {
      cerr << "Internal error creating command line parser" << endl;
      return false;
   }

   // Parse the command line
   if (cl.parseCommandLine(cmdLineArguments) != ParamContainer::errOk) {
      cl.dumpHelp(stderr, true, 78);
      return false;
   }

   if (cl["version"].compare("") != 0) {
      cout << "Git commit ID: " << GIT_COMMIT_ID << endl;
      exit(0);
   }

   // Get the command line values
   Simulator::getInstance().setConfigFileName(cl["configfile"]);
   Simulator::getInstance().setDeserializationFileName(cl["deserializefile"]);
   Simulator::getInstance().setSerializationFileName(cl["serializefile"]);
   Simulator::getInstance().setStimulusFileName(cl["stimulusfile"]);

#if defined(USE_GPU)
   if (EOF == sscanf(cl["device"].c_str(), "%d", &g_deviceId)) {
      g_deviceId = 0;
   }
#endif   // USE_GPU
   return true;
}

///  runSimulation handles command line arguments and loads parameters
///  from parameter file. All initial configuration loading & running the simulator
///  is done here.
///
///  @param  cmdLineArguments command line arguments
///  @param executableName Name of the simultaor's executable file
///  @return -1 if error, else 0 for success.
int Core::runSimulation(string executableName, string cmdLineArguments)
{
   // Get the instance of the console logger and print status
   log4cplus::Logger consoleLogger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("console"));

   LOG4CPLUS_TRACE(consoleLogger, "Instantiating Simulator ");
   Simulator &simulator = Simulator::getInstance();

   LOG4CPLUS_TRACE(consoleLogger, "Instantiating Serializer");
   Serializer serializer;

   // Handles parsing of the command line.
   LOG4CPLUS_TRACE(consoleLogger, "Parsing command line");
   if (!parseCommandLine(executableName, cmdLineArguments)) {
      LOG4CPLUS_FATAL(consoleLogger, "ERROR: failed during command line parse");
      return -1;
   }

   // Loads the configuration file into the Parameter Manager.
   if (!ParameterManager::getInstance().loadParameterFile(simulator.getConfigFileName())) {
      LOG4CPLUS_FATAL(consoleLogger,
                      "ERROR: failed to load config file: " << simulator.getConfigFileName());
      return -1;
   }

   // Read in simulator specific parameters from configuration file.
   LOG4CPLUS_TRACE(consoleLogger, "Loading Simulator parameters");
   simulator.loadParameters();

   // Instantiate simulator objects.
   LOG4CPLUS_TRACE(consoleLogger,
                   "Instantiating Simulator objects specified in configuration file");
   if (!simulator.instantiateSimulatorObjects()) {
      LOG4CPLUS_FATAL(
         consoleLogger,
         "ERROR: Unable to instantiate all simulator classes, check configuration file: "
            + simulator.getConfigFileName() + " for incorrectly declared class types.");
      return -1;
   }

   // Ask all objects to register their Graph properties
   OperationManager::getInstance().executeOperation(Operations::registerGraphProperties);

   // Retrieve class attribute from the 'LayoutParams' in the config file
   // This value indicate the simulation type (Neural or NG911) for graph manager configuration
   // Log fatal error if no simulation type is found and terminate
   string configData;
   ParameterManager::getInstance().getStringByXpath("//LayoutParams/@class", configData);

   if (configData.find("Neur")) {
      GraphManager<NeuralVertexProperties>::getInstance().readGraph();
   }
   if (configData.find("91")) {
      GraphManager<NG911VertexProperties>::getInstance().readGraph();
   } else {
      LOG4CPLUS_FATAL(consoleLogger, "ERROR: Unknown simulation type'");
      return -1;
   }

   // Invoke instantiated simulator objects to load parameters from the configuration file
   LOG4CPLUS_TRACE(consoleLogger, "Loading parameters from configuration file");
   OperationManager::getInstance().executeOperation(Operations::loadParameters);

   // Check if the current user has write permission for the specified serialization path
   if (!simulator.getSerializationFileName().empty()) {
      std::ofstream file(simulator.getSerializationFileName());
      if (file) {
         LOG4CPLUS_TRACE(consoleLogger, "User has write permission for the serialization file.");
      } else {
         LOG4CPLUS_FATAL(consoleLogger,
                         "User does not have write permission for the serialization file.");
         return -1;
      }
   }

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
      if (!serializer.deserialize()) {
         LOG4CPLUS_FATAL(consoleLogger, "Failed while deserializing objects");
         return -1;
      }
   }

   // Helper function for recorder to register spike history variables for all neurons.
   simulator.getModel().getLayout().getVertices().registerHistoryVariables();

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
      serializer.serialize();
   }

   // Tell simulation to clean-up and run any post-simulation logic.
   LOG4CPLUS_TRACE(consoleLogger, "Simulation finished");
   simulator.finish();

   // terminates the simulation recorder
   simulator.getModel().getRecorder().term();


   time(&end_time);
   double timeElapsed = difftime(end_time, start_time);
   double ssps = simulator.getEpochDuration() * simulator.getNumEpochs() / timeElapsed;
   cout << "time simulated: " << simulator.getEpochDuration() * simulator.getNumEpochs() << endl;
   cout << "time elapsed: " << timeElapsed << endl;
   cout << "ssps (simulation seconds / real time seconds): " << ssps << endl;
   return 0;
}
