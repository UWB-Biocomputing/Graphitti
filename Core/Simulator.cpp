/**
 * @file Simulator.cpp
 *
 * @brief Platform independent base class for the Brain Grid simulator.
 * Simulator is a singleton class (a class that can only have one object)
 *
 * @ingroup Core
 */

#include "Simulator.h"
// #include "ParseParamError.h"

// ToDo: Does this need to be a smart ptr?
Simulator *Simulator::instance = nullptr;

/// Acts as constructor, returns the instance of the singleton object
static Simulator *Simulator::getInstance() {
   if (instance == nullptr) {
      instance = Simulator;
   }
   return instance;
};

/// Constructor
Simulator::Simulator() {
   g_simulationStep = 0;  /// uint64_t g_simulationStep instantiated in Global
}

/// Destructor
Simulator::~Simulator() {
   freeResources();
}

/// Initialize and prepare network for simulation.
void Simulator::setup() {
#ifdef PERFORMANCE_METRICS
   // Start overall simulation timer
   cerr << "Starting main timer... ";
   t_host_initialization_layout = 0.0;
   t_host_initialization_connections = 0.0;
   t_host_advance = 0.0;
   t_host_adjustSynapses = 0.0;
   timer.start();
   cerr << "done." << endl;
#endif
   DEBUG(cerr << "Initializing models in network... ";)
   model->setupSim();
   DEBUG(cerr << "\ndone init models." << endl;)
   // init stimulus input object
   if (pInput != NULL) {
      cout << "Initializing input." << endl;
      pInput->init();
   }
}

/// Begin terminating the simulator
void Simulator::finish() {
   model->cleanupSim(); // ToDo: Can #term be removed w/ the new model architecture?  // =>ISIMULATION
}

/// Attempts to read parameters from a XML file.
/// @param  simDoc  the TiXmlDocument to read from.
/// @return true if successful, false otherwise.
bool Simulator::readParameters(TiXmlDocument *simDoc) {
   TiXmlElement *parms = NULL;

   if ((parms = simDoc->FirstChildElement()->FirstChildElement("SimInfoParams")) == NULL) {
      cerr << "Could not find <SimInfoParams> in simulation parameter file " << endl;
      return false;
   }

   try {
      parms->Accept(this);
   } catch (ParseParamError &error) {  // ToDo: is ParseParamError.h necessary after Lizzy's contrib?
      error.print(cerr);
      cerr << endl;
      return false;
   }

   // check to see if all required parameters were successfully read
   if (checkNumParameters() != true) {
      cerr << "Some parameters are missing in <SimInfoParams> in simulation parameter file " << endl;
      return false;
   }

   return true;
}

/// Handles loading of parameters using tinyxml from the parameter file.
/// @param  element TiXmlElement to examine.
/// @param  firstAttribute  ***NOT USED***.
/// @return true if method finishes without errors.
bool Simulator::VisitEnter(const TiXmlElement &element, const TiXmlAttribute *firstAttribute)
//TODO: firstAttribute does not seem to be used! Delete?
{
   static string parentNode = "";
   if (element.ValueStr().compare("SimInfoParams") == 0) {
      return true;
   }

   if (element.ValueStr().compare("PoolSize") == 0 ||
       element.ValueStr().compare("SimParams") == 0 ||
       element.ValueStr().compare("SimConfig") == 0 ||
       element.ValueStr().compare("Seed") == 0 ||
       element.ValueStr().compare("OutputParams") == 0) {
      nParams++;                                                                               // where is first value of nparams being sourced from? declared but cant find initial value.

      return true;
   }

   if (element.Parent()->ValueStr().compare("PoolSize") == 0) {
      if (element.ValueStr().compare("x") == 0) {
         width = atoi(element.GetText());
      } else if (element.ValueStr().compare("y") == 0) {
         height = atoi(element.GetText());
      }

      if (width != 0 && height != 0) {
         totalNeurons = width * height;
      }
      return true;
   }

   if (element.Parent()->ValueStr().compare("SimParams") == 0) {

      if (element.ValueStr().compare("Tsim") == 0) {
         epochDuration = atof(element.GetText());
      } else if (element.ValueStr().compare("numSims") == 0) {
         maxSteps = atof(element.GetText());
      }

      if (epochDuration < 0 || maxSteps < 0) {
         throw ParseParamError("SimParams", "Invalid negative SimParams value.");
      }

      return true;
   }

   if (element.Parent()->ValueStr().compare("SimConfig") == 0) {
      if (element.ValueStr().compare("maxFiringRate") == 0) {
         maxFiringRate = atoi(element.GetText());
      } else if (element.ValueStr().compare("maxSynapsesPerNeuron") == 0) {
         maxSynapsesPerNeuron = atoi(element.GetText());
      }

      if (maxFiringRate < 0 || maxSynapsesPerNeuron < 0) {
         throw ParseParamError("SimConfig", "Invalid negative SimConfig value.");
      }

      return true;
   }

   if (element.Parent()->ValueStr().compare("Seed") == 0) {
      if (element.ValueStr().compare("value") == 0) {
         seed = atoi(element.GetText());
      }
      return true;
   }

   if (element.Parent()->ValueStr().compare("OutputParams") == 0) {
      // file name specified in command line is higher priority

      if (stateOutputFileName.empty()) {
         if (element.ValueStr().compare("stateOutputFileName") == 0) {
            stateOutputFileName = element.GetText();
         }
      }
      return true;
   }

   return false;
}

/// Prints out loaded parameters to ostream.
/// @param  output  ostream to send output to.
void Simulator::printParameters(ostream &output) const {
   cout << "poolsize x:" << width << " y:" << height
        << endl;
   cout << "Simulation Parameters:\n";
   cout << "\tTime between growth updates (in seconds): " << epochDuration << endl;
   cout << "\tNumber of simulations to run: " << maxSteps << endl;
}


/// Copy GPU Synapse data to CPU.
void Simulator::copyGPUSynapseToCPU() {
   model->copyGPUSynapseToCPUModel();
}

/// Copy CPU Synapse data to GPU.
void Simulator::copyCPUSynapseToGPU() {
   model->copyCPUSynapseToGPUModel();
}

/// Resets all of the maps. Releases and re-allocates memory for each map, clearing them as necessary.
void Simulator::reset() {
   DEBUG(cout << "\nEntering Simulator::reset()" << endl;)
   // Terminate the simulator
   model->cleanupSim();
   // Clean up objects
   freeResources();
   // Reset global simulation Step to 0
   g_simulationStep = 0;
   // Initialize and prepare network for simulation
   model->setupSim();
   DEBUG(cout << "\nExiting Simulator::reset()" << endl;)
}

/// Clean up objects.
void Simulator::freeResources() {}

/// Run simulation
void Simulator::simulate() {
   // Main simulation loop - execute maxGrowthSteps
   for (int currentStep = 1; currentStep <= maxSteps; currentStep++) {
      DEBUG(cout << endl << endl;)
      DEBUG(cout << "Performing simulation number " << currentStep << endl;)
      DEBUG(cout << "Begin network state:" << endl;)
      // Init SimulationInfo parameters
      currentStep = currentStep;
#ifdef PERFORMANCE_METRICS
      // Start timer for advance
      short_timer.start();
#endif
      // Advance simulation to next growth cycle
      advanceUntilGrowth(currentStep);
#ifdef PERFORMANCE_METRICS
      // Time to advance
      t_host_advance += short_timer.lap() / 1000000.0;
#endif
      DEBUG(cout << endl << endl;)
      DEBUG(
            cout << "Done with simulation cycle, beginning growth update "
                 << currentStep << endl;
      )
      // Update the neuron network

#ifdef PERFORMANCE_METRICS
      // Start timer for connection update
      short_timer.start();
#endif

      model->updateConnections();
      model->updateHistory();

#ifdef PERFORMANCE_METRICS
      // Times converted from microseconds to seconds
      // Time to update synapses
      t_host_adjustSynapses += short_timer.lap() / 1000000.0;
      // Time since start of simulation
      double total_time = timer.lap() / 1000000.0;

      cout << "\ntotal_time: " << total_time << " seconds" << endl;
      printPerformanceMetrics(total_time, currentStep);
      cout << endl;
#endif
   }
}

/// Helper for #simulate(). Advance simulation until ready for next growth cycle.
/// This should simulate all neuron and synapse activity for one epoch.
/// @param currentStep the current epoch in which the network is being simulated.
void Simulator::advanceUntilGrowth(const int currentStep) {
   uint64_t count = 0;
   // Compute step number at end of this simulation epoch
   uint64_t endStep = g_simulationStep
                      + static_cast<uint64_t>(epochDuration / deltaT);
   DEBUG_MID(model->logSimStep();) // Generic model debug call
   while (g_simulationStep < endStep) {
      DEBUG_LOW(
      // Output status once every 10,000 steps
      if (count % 10000 == 0) {
         cout << currentStep << "/" << maxSteps
              << " simulating time: "
              << g_simulationStep * deltaT << endl;
         count = 0;
      }
      count++;
      )
      // input stimulus
      if (pInput != NULL)
         pInput->inputStimulus();
      // Advance the Network one time step
      model->advance();
      g_simulationStep++;
   }
}

/// Writes simulation results to an output destination.
void Simulator::saveData() const {
   model->saveData();
}

/************************************************
 *  Mutators
 ***********************************************/

/// List of summation points (either host or device memory)
void Simulator::setPSummationMap(BGFLOAT *summationMap) {
   pSummationMap = summationMap;
}

/************************************************
 *  Accessors
 ***********************************************/

int Simulator::getWidth() const { return width; }

int Simulator::getHeight() const { return height; }

int Simulator::getTotalNeurons() const { return totalNeurons; }

int Simulator::getCurrentStep() const { return currentStep; }

int Simulator::getMaxSteps() const { return maxSteps; }

BGFLOAT Simulator::getEpochDuration() const { return epochDuration; }

int Simulator::getMaxFiringRate() const { return maxFiringRate; } /// **GPU Only**

int Simulator::getMaxSynapsesPerNeuron() const { return maxSynapsesPerNeuron; } ///  **GPU Only.**

BGFLOAT Simulator::getDeltaT() const { return deltaT; }

// ToDo: should be a vector of neuron type
// ToDo: vector should be contiguous array, resize is used.
neuronType *Simulator::getRgNeuronTypeMap() const { return rgNeuronTypeMap; }

// ToDo: make smart ptr
/// Starter existence map (T/F).
bool *Simulator::getRgEndogenouslyActiveNeuronMap() const { return rgEndogenouslyActiveNeuronMap; }

BGFLOAT Simulator::getMaxRate() const { return maxRate; } // TODO: more detail here

BGFLOAT *Simulator::getPSummationMap() const { return pSummationMap; }

long Simulator::getSeed() const { return seed; }

string Simulator::getStateOutputFileName() const { return stateOutputFileName; }

string Simulator::getStateInputFileName() const { return stateInputFileName; }

string Simulator::getMemOutputFileName() const { return memOutputFileName; }

string Simulator::getMemInputFileName() const { return memInputFileName; }

string Simulator::getStimulusInputFileName() const { return stimulusInputFileName; }

IModel *Simulator::getModel() const { return model; } /// ToDo: make smart ptr

IRecorder *Simulator::getSimRecorder() const { return simRecorder; } /// ToDo: make smart ptr

ISInput *Simulator::getPInput() const { return pInput; } /// ToDo: make smart ptr

Timer Simulator::getTimer() const { return timer; }

Timer Simulator::getShort_timer() const { return short_timer; }


