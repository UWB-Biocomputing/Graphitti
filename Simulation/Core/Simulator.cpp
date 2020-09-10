/**
 * @file Simulator.cpp
 *
 * @brief Platform independent base class for the Brain Grid simulator.
 * Simulator is a singleton class (a class that can only have one object)
 *
 * @ingroup Core
 */

#include "Simulator.h"

#include <functional>

#include "CPUSpikingModel.h"
#include "OperationManager.h"
#include "ParameterManager.h"
#include "RecorderFactory.h"
// #include "ParseParamError.h"

/// Acts as constructor first time it's called, returns the instance of the singleton object
Simulator &Simulator::getInstance() {
   static Simulator instance;
   return instance;
};

/// Constructor is private to keep a singleton instance of this class.
Simulator::Simulator() {
   g_simulationStep = 0;  /// uint64_t g_simulationStep instantiated in Global
   deltaT_ = DEFAULT_dt;

   // Register printParameters function as a printParameters operation in the OperationManager
   function<void()> printParametersFunc = bind(&Simulator::printParameters, this);
   OperationManager::getInstance().registerOperation(Operations::printParameters, printParametersFunc);
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
   model_->setupSim();
   DEBUG(cerr << "\ndone init models." << endl;)

   // init stimulus input object
   /* PInput not in project yet
   if (pInput != NULL) {
      cout << "Initializing input." << endl;
      pInput->init();
   }
    */
}

/// Begin terminating the simulator
void Simulator::finish() {
   model_->cleanupSim(); // ToDo: Can #term be removed w/ the new model architecture?  // =>ISIMULATION
}

/// Load member variables from configuration file
void Simulator::loadParameters() {
   ParameterManager::getInstance().getIntByXpath("//PoolSize/x/text()", width_);
   ParameterManager::getInstance().getIntByXpath("//PoolSize/y/text()", height_);
   totalNeurons_ = width_ * height_;

   ParameterManager::getInstance().getBGFloatByXpath("//SimParams/epochDuration/text()", epochDuration_);
   ParameterManager::getInstance().getIntByXpath("//SimParams/numEpochs/text()", numEpochs_);
   ParameterManager::getInstance().getIntByXpath("//SimConfig/maxFiringRate/text()", maxFiringRate_);
   ParameterManager::getInstance().getIntByXpath("//SimConfig/maxSynapsesPerNeuron/text()", maxSynapsesPerNeuron_);
   ParameterManager::getInstance().getLongByXpath("//Seed/value/text()", seed_);

   // Result file name can be set by the command line arguments so check for default string value as to not overwrite it
   if (resultFileName_ == "") {
      ParameterManager::getInstance().getStringByXpath("//OutputParams/resultFileName/text()", resultFileName_);
   }
}

/// Prints out loaded parameters to console.
void Simulator::printParameters() const {
   cout << "SIMULATION PARAMETERS" << endl;
   cout << "\tpoolsize x:" << width_ << " y:" << height_
        << endl;
   cout << "\tTime between growth updates (in seconds): " << epochDuration_ << endl;
   cout << "\tNumber of epochs to run: " << numEpochs_ << endl;
   cout << "\tMax firing rate: " << maxFiringRate_ << endl;
   cout << "\tMax synapses per neuron: " << maxSynapsesPerNeuron_ << endl;
   cout << "\tSeed: " << seed_ << endl;
   cout << "\tResult file path: " << resultFileName_ << endl << endl;
}


/// Copy GPU Synapse data to CPU.
void Simulator::copyGPUSynapseToCPU() {
   // ToDo: Delete this method and implement using OperationManager
   // model->copyGPUSynapseToCPUModel();
}

/// Copy CPU Synapse data to GPU.
void Simulator::copyCPUSynapseToGPU() {
   // ToDo: Delete this method and implement using OperationManager
   // model->copyCPUSynapseToGPUModel();
}

/// Resets all of the maps. Releases and re-allocates memory for each map, clearing them as necessary.
void Simulator::reset() {
   DEBUG(cout << "\nEntering Simulator::reset()" << endl;)
   // Terminate the simulator
   model_->cleanupSim();
   // Clean up objects
   freeResources();
   // Reset global simulation Step to 0
   g_simulationStep = 0;
   // Initialize and prepare network for simulation
   model_->setupSim();
   DEBUG(cout << "\nExiting Simulator::reset()" << endl;)
}

/// Clean up objects.
void Simulator::freeResources() {}

/// Run simulation
void Simulator::simulate() {
   // Main simulation loop - execute maxGrowthSteps
   for (int currentEpoch = 1; currentEpoch <= numEpochs_; currentEpoch++) {
      DEBUG(cout << endl << endl;)
      DEBUG(cout << "Performing simulation number " << currentEpoch << endl;)
      DEBUG(cout << "Begin network state:" << endl;)
      // Init SimulationInfo parameters
      currentEpoch_ = currentEpoch;
#ifdef PERFORMANCE_METRICS
      // Start timer for advance
      short_timer.start();
#endif
      // Advance simulation to next growth cycle
      advanceUntilGrowth(currentEpoch);
#ifdef PERFORMANCE_METRICS
      // Time to advance
      t_host_advance += short_timer.lap() / 1000000.0;
#endif
      DEBUG(cout << endl << endl;)
      DEBUG(
            cout << "Done with simulation cycle, beginning growth update "
                 << currentEpoch << endl;
      )
      // Update the neuron network

#ifdef PERFORMANCE_METRICS
      // Start timer for connection update
      short_timer.start();
#endif

      model_->updateConnections();
      model_->updateHistory();

#ifdef PERFORMANCE_METRICS
      // Times converted from microseconds to seconds
      // Time to update synapses
      t_host_adjustSynapses += short_timer.lap() / 1000000.0;
      // Time since start of simulation
      double total_time = timer.lap() / 1000000.0;

      cout << "\ntotal_time: " << total_time << " seconds" << endl;
      printPerformanceMetrics(total_time, currentEpoch);
      cout << endl;
#endif
   }
}

/// Helper for #simulate(). Advance simulation until ready for next growth cycle.
/// This should simulate all neuron and synapse activity for one epoch.
/// @param currentStep the current epoch in which the network is being simulated.
void Simulator::advanceUntilGrowth(const int &currentEpoch) const {
   uint64_t count = 0;
   // Compute step number at end of this simulation epoch
   uint64_t endStep = g_simulationStep
                      + static_cast<uint64_t>(epochDuration_ / deltaT_);
   // DEBUG_MID(model->logSimStep();) // Generic model debug call
   while (g_simulationStep < endStep) {
      DEBUG_LOW(
      // Output status once every 10,000 steps
      if (count % 10000 == 0) {
         cout << currentEpoch << "/" << numEpochs_
              << " simulating time: "
              << g_simulationStep * deltaT_ << endl;
         count = 0;
      }
      count++;
      )
      // input stimulus
      /***** S_INPUT NOT IN REPO YET *******/
//      if (pInput != NULL)
//         pInput->inputStimulus();
      // Advance the Network one time step
      model_->advance();
      g_simulationStep++;
   }
}

/// Writes simulation results to an output destination.
void Simulator::saveData() const {
   model_->saveData();
}

/// Instantiates Model which causes all other lower level simulator objects to be instantiated. Checks if all
/// expected objects were created correctly and returns T/F on the success of the check.
bool Simulator::instantiateSimulatorObjects() {
   // Model Definition
#ifdef USE_GPU
   model_ = shared_ptr<Model>(new GPUSpikingModel());
#else
   model_ = shared_ptr<Model>(new CPUSpikingModel());
#endif

   // Perform check on all instantiated objects.
   if (!model_
   || !model_->getConnections()
   || !model_->getConnections()->getSynapses()
   || !model_->getLayout()
   || !model_->getLayout()->getNeurons()
   || !model_->getRecorder()) {
      return false;
   }
   return true;
}


/************************************************
 *  Mutators
 ***********************************************/

/// List of summation points (either host or device memory)
void Simulator::setPSummationMap(BGFLOAT *summationMap) { pSummationMap_ = summationMap; }

void Simulator::setResultFileName(const string &fileName) { resultFileName_ = fileName; }

void Simulator::setConfigFileName(const string &fileName) { configFileName_ = fileName; }

void Simulator::setSerializationFileName(const string &fileName) { serializationFileName_ = fileName; }

void Simulator::setDeserializationFileName(const string &fileName) { deserializationFileName_ = fileName; }

void Simulator::setStimulusFileName(const string &fileName) { stimulusFileName_ = fileName; }

/************************************************
 *  Accessors
 ***********************************************/

int Simulator::getWidth() const { return width_; }

int Simulator::getHeight() const { return height_; }

int Simulator::getTotalNeurons() const { return totalNeurons_; }

int Simulator::getCurrentStep() const { return currentEpoch_; }

int Simulator::getNumEpochs() const { return numEpochs_; }

BGFLOAT Simulator::getEpochDuration() const { return epochDuration_; }

int Simulator::getMaxFiringRate() const { return maxFiringRate_; } /// **GPU Only**

int Simulator::getMaxSynapsesPerNeuron() const { return maxSynapsesPerNeuron_; } ///  **GPU Only.**

BGFLOAT Simulator::getDeltaT() const { return deltaT_; }

// ToDo: should be a vector of neuron type
// ToDo: vector should be contiguous array, resize is used.
neuronType *Simulator::getRgNeuronTypeMap() const { return rgNeuronTypeMap_; }

// ToDo: make smart ptr
/// Starter existence map (T/F).
bool *Simulator::getRgEndogenouslyActiveNeuronMap() const { return rgEndogenouslyActiveNeuronMap_; }

BGFLOAT Simulator::getMaxRate() const { return maxRate_; }

BGFLOAT *Simulator::getPSummationMap() const { return pSummationMap_; }

long Simulator::getSeed() const { return seed_; }

string Simulator::getResultFileName() const { return resultFileName_; }

string Simulator::getConfigFileName() const { return configFileName_; }

string Simulator::getSerializationFileName() const { return serializationFileName_; }

string Simulator::getDeserializationFileName() const { return deserializationFileName_; }

string Simulator::getStimulusFileName() const { return stimulusFileName_; }

shared_ptr<Model> Simulator::getModel() const { return model_; }

#ifdef PERFOMANCE_METRICS
Timer Simulator::getTimer() const { return timer; }

Timer Simulator::getShort_timer() const { return short_timer; }
#endif




