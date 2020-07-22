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

/// Acts as constructor, returns the instance of the singleton object
Simulator &Simulator::getInstance() {
   static Simulator instance;
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
void Simulator::advanceUntilGrowth(const int &currentStep) const {
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
      /***** S_INPUT NOT IN REPO YET *******/
//      if (pInput != NULL)
//         pInput->inputStimulus();
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

#ifdef PERFOMANCE_METRICS
Timer Simulator::getTimer() const { return timer; }

Timer Simulator::getShort_timer() const { return short_timer; }
#endif

BGFLOAT Simulator::getMaxRate() const { return maxRate; } // TODO: more detail here


