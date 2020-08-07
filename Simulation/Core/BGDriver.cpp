/**
 * @file Driver
 *
 *  The driver performs the following steps:
 *  1) reads parameters from an xml file (specified as the first argument)
 *  2) creates the network
 *  3) launches the simulation
 *
 */

#include <fstream>
#include "Global.h"
#include "ThirdParty/paramcontainer/ParamContainer.h"

#include "Model.h"
#include "IRecorder.h"
#include "Utils/Inputs/FSInput.h"
#include "Simulator.h"
#include "ParameterManager.h"
#include "OperationManager.h"
#include "VerticiesFactory.h"

// Uncomment to use visual leak detector (Visual Studios Plugin)
// #include <vld.h>


//! Cereal
//#include <cereal/archives/xml.hpp>
//#include <cereal/archives/binary.hpp>
//#include "ConnGrowth.h" // hacked in. that's why its here.

#if defined(USE_GPU)
#include "GPUSpikingModel.h"
#elif defined(USE_OMP)
//    #include "MultiThreadedSim.h"
#else

#include "CPUSpikingModel.h"

#endif

using namespace std;

// functions
bool parseCommandLine(int argc, char *argv[]);

void instantiateSimulationObjects();

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
   Simulator &simulator = Simulator::getInstance();

   // Handles parsing of the command line.
   if (!parseCommandLine(argc, argv)) {
      cerr << "ERROR: failed during command line parse" << endl;
      return -1;
   }

   // Loads the configuration file into the Parameter Manager.
   if (!ParameterManager::getInstance().loadParameterFile(Simulator::getInstance().getParameterFileName())) {
      cerr << "ERROR: failed to load config file: " << Simulator::getInstance().getParameterFileName() << endl;
      return -1;
   }

   // Read in global parameters from configuration file
   simulator.loadParameters();
   simulator.printParameters();

   // Instantiate simulator objects
   if (!simulator.instantiateSimulatorObjects()) {
      cerr << "ERROR: Unable to instantiate all simulator classes, check configuration file: "
           << simulator.getParameterFileName()
           << " for incorrectly declared class types." << endl;
      return -1;
   }

   // Setup class ownership and initialize each classes parameters using parameter manager

   // Create all model instances and load parameters from a file.
   // todo: parameter manager replaces this - parses config file
   // todo: all readparams methods in lower level classes are going to look very different.
   // objects will be calling param manager asking for their own things .

   // two phase process: global params, and then obj init.
//   if (!LoadAllParameters()) {
//      cerr << "! ERROR: failed while parsing simulation parameters." << endl;
//      return -1;
//   }
//
//   // todo: this should be a job of the simulator. replaced by a call to simulator
//
//   // create & init simulation recorder
//   simInfo->simRecorder = simInfo->model->getConnections()->createRecorder();
//   if (simInfo->simRecorder == NULL) {
//      cerr << "! ERROR: invalid state output file name extension." << endl;
//      return -1;
//   }
//
//   // Create a stimulus input object
//   simInfo->pInput = FSInput::get()->CreateInstance();
//
//   time_t start_time, end_time;
//   time(&start_time);
//
//   // in chain of responsibility. still going to exist!
//   // setup simulation
//   DEBUG(cerr << "Setup simulation." << endl;)
//   simulator->setup();
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
//   {
//      simInfo->pInput->term();
//      delete simInfo->pInput;
//   }
//
//   // todo: before this, do copy from gpu.
//   // Writes simulation results to an output destination
//   simulator->saveData();
//
//   // todo: going to be moved. with the "hack"
//   // Serializes internal state for the current simulation
//   if (!simInfo->memOutputFileName.empty()) {
//
//      // Serialization
//      serializeSynapseInfo();
//
//      DEBUG(
//      // Prints out internal state information after serialization
//            cout << "------------------------------After Serialization:------------------------------" << endl;
//            printKeyStateInfo();
//      )
//
//   }
//
//   // todo: handled by chain of responsibility for termination/cleanup.
//   // Tell simulation to clean-up and run any post-simulation logic.
//   simulator->finish();
//
//   // terminates the simulation recorder
//   if (simInfo->simRecorder != NULL) {
//      simInfo->simRecorder->term();
//   }
//
//   for(unsigned int i = 0; i < rgNormrnd.size(); ++i) {
//      delete rgNormrnd[i];
//   }
//
//   rgNormrnd.clear();
//
//   time(&end_time);
//   double time_elapsed = difftime(end_time, start_time);
//   double ssps = simInfo->epochDuration * simInfo->maxSteps / time_elapsed;
//   cout << "time simulated: " << simInfo->epochDuration * simInfo->maxSteps << endl;
//   cout << "time elapsed: " << time_elapsed << endl;
//   cout << "ssps (simulation seconds / real time seconds): " << ssps << endl;
//
//   delete simInfo->model;
//   simInfo->model = NULL;
//
//   if (simInfo->simRecorder != NULL) {
//      delete simInfo->simRecorder;
//      simInfo->simRecorder = NULL;
//   }
//
//   delete simInfo;
//   simInfo = NULL;
//
//   delete simulator;
//   simulator = NULL;

   return 0;
}

bool parseCommandLine(int argc, char *argv[]) {
   ParamContainer cl; // todo: note as third party class.
   cl.initOptions(false);  // don't allow unknown parameters
   cl.setHelpString(string("The DCT growth modeling simulator\nUsage: ") + argv[0] + " ");

#if defined(USE_GPU)
   if ((cl.addParam("stateoutfile", 'o', ParamContainer::filename, "simulation results filepath") != ParamContainer::errOk)
            || (cl.addParam("resultfile", 't', ParamContainer::filename | ParamContainer::required, "parameter configuration filepath") != ParamContainer::errOk)
            || (cl.addParam("paramfile", 'd', ParamContainer::regular, "CUDA device id") != ParamContainer::errOk)
            || (cl.addParam( "stimulusfile", 's', ParamContainer::filename, "stimulus input filepath" ) != ParamContainer::errOk)
            || (cl.addParam("meminfile", 'r', ParamContainer::filename, "simulation memory image input filepath") != ParamContainer::errOk)
            || (cl.addParam("memoutfile", 'w', ParamContainer::filename, "simulation memory image output filepath") != ParamContainer::errOk)) {
        cerr << "Internal error creating command line parser" << endl;
        return false;
    }
#else    // !USE_GPU
   if ((cl.addParam("resultfile", 'o', ParamContainer::filename, "simulation results filepath") !=
        ParamContainer::errOk)
       || (cl.addParam("paramfile", 't', ParamContainer::filename | ParamContainer::required,
                       "parameter configuration filepath") != ParamContainer::errOk)
       ||
       (cl.addParam("stimulusfile", 's', ParamContainer::filename, "stimulus input filepath") != ParamContainer::errOk)
       || (cl.addParam("meminfile", 'r', ParamContainer::filename, "simulation memory image filepath") !=
           ParamContainer::errOk)
       || (cl.addParam("memoutfile", 'w', ParamContainer::filename, "simulation memory image output filepath") !=
           ParamContainer::errOk)) {
      cerr << "Internal error creating command line parser" << endl;
      return false;
   }
#endif  // USE_GPU

   // Parse the command line
   if (cl.parseCommandLine(argc, argv) != ParamContainer::errOk) {
      cl.dumpHelp(stderr, true, 78);
      return false;
   }

   // Get the command line values
   Simulator::getInstance().setResultFileName(cl["resultfile"]);
   Simulator::getInstance().setParameterFileName(cl["paramfile"]);
   Simulator::getInstance().setMemInputFileName(cl["meminfile"]);
   Simulator::getInstance().setMemOutputFileName(cl["memoutfile"]);
   Simulator::getInstance().setStimulusFileName(cl["stimulusfile"]);

#if defined(USE_GPU)
   if (EOF == sscanf(cl["deviceid"].c_str(), "%d", &g_deviceId)) {
       g_deviceId = 0;
   }
#endif  // USE_GPU
   return true;
}