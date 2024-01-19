/**
 * @file Simulator.cpp
 *
 * @brief Platform independent base class for the Brain Grid simulator.
 * Simulator is a singleton class (a class that can only have one object)
 *
 * @ingroup Simulator/Core
 */

#include "Simulator.h"
#include "CPUModel.h"
#include "GPUModel.h"
#include "OperationManager.h"
#include "ParameterManager.h"
#include "Utils/Factory.h"
#include <functional>
// #include "ParseParamError.h"

/// Acts as constructor first time it's called, returns the instance of the
/// singleton object
Simulator &Simulator::getInstance()
{
   static Simulator instance;
   return instance;
};

/// Constructor is private to keep a singleton instance of this class.
Simulator::Simulator()
{
   g_simulationStep = 0;   /// uint64_t g_simulationStep instantiated in Global

   consoleLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("console"));
   fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));
   edgeLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("edge"));
   workbenchLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("workbench"));

   // Register printParameters function as a printParameters operation in the OperationManager
   function<void()> printParametersFunc = bind(&Simulator::printParameters, this);
   OperationManager::getInstance().registerOperation(Operations::printParameters,
                                                     printParametersFunc);
}

/// Initialize and prepare network for simulation.
void Simulator::setup()
{
#ifdef PERFORMANCE_METRICS
   // Start overall simulation timer
   cerr << "Starting main timer... ";
   t_host_initialization_layout = 0.0;
   t_host_initialization_connections = 0.0;
   t_host_advance = 0.0;
   t_host_adjustEdges = 0.0;
   t_host_updateHistory = 0.0;
   timer.start();
   cerr << "done." << endl;
#endif
   LOG4CPLUS_INFO(fileLogger_, "Initializing models in network... ");
   model_->setupSim();
   LOG4CPLUS_INFO(fileLogger_, "Model initialization finished");

   // init stimulus input object
   /* PInput not in project yet
  if (pInput != nullptr) {
     cout << "Initializing input." << endl;
     pInput->init();
  }
  */
}

/// Begin terminating the simulator
void Simulator::finish()
{
   model_->finish();   // ToDo: Can #term be removed w/ the new model architecture?
                       // // =>ISIMULATION
}

/// Load member variables from configuration file
void Simulator::loadParameters()
{
   ParameterManager::getInstance().getBGFloatByXpath("//SimParams/epochDuration/text()",
                                                     epochDuration_);
   ParameterManager::getInstance().getIntByXpath("//SimParams/numEpochs/text()", numEpochs_);
   ParameterManager::getInstance().getIntByXpath("//SimConfig/maxFiringRate/text()",
                                                 maxFiringRate_);
   ParameterManager::getInstance().getIntByXpath("//SimConfig/maxEdgesPerVertex/text()",
                                                 maxEdgesPerVertex_);

   // Use default deltaT_ if not present in config file
   if (!ParameterManager::getInstance().getBGFloatByXpath("//SimParams/deltaT/text()", deltaT_)) {
      deltaT_ = DEFAULT_dt;
   }

   // Instantiate rng object
   string type;
   ParameterManager::getInstance().getStringByXpath("//RNGConfig/NoiseRNGSeed/@class", type);
   noiseRNG = Factory<MTRand>::getInstance().createType(type);

   ParameterManager::getInstance().getLongByXpath("//RNGConfig/InitRNGSeed/text()", initRngSeed_);
   ParameterManager::getInstance().getLongByXpath("//RNGConfig/NoiseRNGSeed/text()", noiseRngSeed_);
   noiseRNG->seed(noiseRngSeed_);
   initRNG.seed(initRngSeed_);
}

/// Prints out loaded parameters to logging file.
void Simulator::printParameters() const
{
   LOG4CPLUS_DEBUG(fileLogger_,
                   "\nSIMULATION PARAMETERS"
                      << endl
                      << "\tTime between growth updates (in seconds): " << epochDuration_ << endl
                      << "\tNumber of epochs to run: " << numEpochs_ << endl
                      << "\tMax firing rate: " << maxFiringRate_ << endl
                      << "\tMax edges per vertex: " << maxEdgesPerVertex_ << endl
                      << "\tNoise RNG Seed: " << noiseRngSeed_ << endl
                      << "\tInitializer RNG Seed: " << initRngSeed_ << endl);
}

// Code from STDPFix branch, doesn't do anything
/// Copy GPU Synapse data to CPU.
void Simulator::copyGPUSynapseToCPU()
{
   // ToDo: Delete this method and implement using OperationManager
   // model->copyGPUSynapseToCPUModel();
}

/// Copy CPU Synapse data to GPU.
void Simulator::copyCPUSynapseToGPU()
{
   // ToDo: Delete this method and implement using OperationManager
   // model->copyCPUSynapseToGPUModel();
}

/// Resets all of the maps. Releases and re-allocates memory for each map,
/// clearing them as necessary.
void Simulator::reset()
{
   LOG4CPLUS_INFO(fileLogger_, "Resetting Simulator");
   // Terminate the simulator
   model_->finish();
   // Reset global simulation Step to 0
   g_simulationStep = 0;
   // Initialize and prepare network for simulation
   model_->setupSim();
   LOG4CPLUS_INFO(fileLogger_, "Simulator Reset Finished");
}

/// Run simulation
void Simulator::simulate()
{
   // Main simulation loop - execute maxGrowthSteps
   for (int currentEpoch = 1; currentEpoch <= numEpochs_; currentEpoch++) {
      LOG4CPLUS_TRACE(consoleLogger_, "Performing epoch number: " << currentEpoch);
      LOG4CPLUS_TRACE(fileLogger_, "Begin network state:");
      currentEpoch_ = currentEpoch;
#ifdef PERFORMANCE_METRICS
      // Start timer for advance
      short_timer.start();
#endif
      // Advance simulation to next growth cycle
      advanceEpoch(currentEpoch);
#ifdef PERFORMANCE_METRICS
      // Time to advance
      t_host_advance += short_timer.lap() / 1000000.0;
#endif
      LOG4CPLUS_TRACE(consoleLogger_,
                      "done with epoch cycle " << currentEpoch_ << ", beginning growth update");
      LOG4CPLUS_TRACE(edgeLogger_, "Epoch: " << currentEpoch_);

#ifdef PERFORMANCE_METRICS
      // Start timer for connection update
      short_timer.start();
#endif
      // Update the neuron network
      model_->updateConnections();

#ifdef PERFORMANCE_METRICS
      // Times converted from microseconds to seconds
      // Time to update synapses
      t_host_adjustEdges += short_timer.lap() / 1000000.0;
      // Start timer for history update
      short_timer.start();

#endif
      // Update the neuron network
      model_->updateHistory();

#ifdef PERFORMANCE_METRICS
      // Times converted from microseconds to seconds
      // Time to update history
      t_host_updateHistory += short_timer.lap() / 1000000.0;
      // Time since start of simulation
      double total_time = timer.lap() / 1000000.0;

      cout << "\ntotal_time: " << total_time << " seconds" << endl;
      printPerformanceMetrics(total_time, currentEpoch);
      cout << endl;
#endif
   }
   LOG4CPLUS_TRACE(workbenchLogger_, "Complete");
}

/// Helper for #simulate(). Advance simulation until ready for next growth cycle.
/// This should simulate all neuron and synapse activity for one epoch.
/// @param currentStep the current epoch in which the network is being simulated.
void Simulator::advanceEpoch(int currentEpoch) const
{
   uint64_t count = 0;
   // Compute step number at end of this simulation epoch
   uint64_t endStep = g_simulationStep + static_cast<uint64_t>(epochDuration_ / deltaT_);
   model_->getLayout().getVertices().loadEpochInputs(g_simulationStep, endStep);
   // DEBUG_MID(model->logSimStep();) // Generic model debug call
   uint64_t onePercent = (epochDuration_ / deltaT_) * numEpochs_ * 0.01;
   while (g_simulationStep < endStep) {
      // Output status once every 1% of total simulation time
      if (count % onePercent == 0) {
         LOG4CPLUS_TRACE(consoleLogger_,
                         "Epoch: " << currentEpoch << "/" << numEpochs_
                                   << " simulating time: " << g_simulationStep * deltaT_ << "/"
                                   << (epochDuration_ * numEpochs_) - 1);
         LOG4CPLUS_TRACE(workbenchLogger_,
                         "Epoch: " << currentEpoch << "/" << numEpochs_
                                   << " simulating time: " << g_simulationStep * deltaT_ << "/"
                                   << (epochDuration_ * numEpochs_) - 1);
         count = 0;
      }
      count++;
      // input stimulus
      /***** S_INPUT NOT IN REPO YET *******/
      //      if (pInput != nullptr)
      //         pInput->inputStimulus();
      // Advance the Network one time step
      model_->advance();
      g_simulationStep++;
   }
}

/// Writes simulation results to an output destination.
void Simulator::saveResults() const
{
   model_->saveResults();
}

/// Instantiates Model which causes all other lower level simulator objects to
/// be instantiated. Checks if all expected objects were created correctly and
/// returns T/F on the success of the check.
bool Simulator::instantiateSimulatorObjects()
{
   // Model Definition
#if defined(USE_GPU)
   model_ = unique_ptr<Model>(new GPUModel());
#else
   model_ = unique_ptr<Model>(new CPUModel());
#endif

   // Perform check on all instantiated objects.
   if (!model_) {
      return false;
   }
   return true;
}

/************************************************
 *  Mutators
 ***********************************************/
///@{

void Simulator::setConfigFileName(const string &fileName)
{
   configFileName_ = fileName;
}

void Simulator::setSerializationFileName(const string &fileName)
{
   serializationFileName_ = fileName;
}

void Simulator::setDeserializationFileName(const string &fileName)
{
   deserializationFileName_ = fileName;
}

void Simulator::setStimulusFileName(const string &fileName)
{
   stimulusFileName_ = fileName;
}

///@}

/************************************************
 *  Accessors
 ***********************************************/
///@{
int Simulator::getTotalVertices() const
{
   return model_->getLayout().getNumVertices();
}

int Simulator::getCurrentStep() const
{
   return currentEpoch_;
}

int Simulator::getNumEpochs() const
{
   return numEpochs_;
}

BGFLOAT Simulator::getEpochDuration() const
{
   return epochDuration_;
}

int Simulator::getMaxFiringRate() const
{
   return maxFiringRate_;
}   /// **GPU Only**

int Simulator::getMaxEdgesPerVertex() const
{
   return maxEdgesPerVertex_;
}   ///  **GPU Only.**

BGFLOAT Simulator::getDeltaT() const
{
   return deltaT_;
}

BGFLOAT Simulator::getMaxRate() const
{
   return maxRate_;
}

long Simulator::getNoiseRngSeed() const
{
   return noiseRngSeed_;
}

long Simulator::getInitRngSeed() const
{
   return initRngSeed_;
}

string Simulator::getConfigFileName() const
{
   return configFileName_;
}

string Simulator::getSerializationFileName() const
{
   return serializationFileName_;
}

string Simulator::getDeserializationFileName() const
{
   return deserializationFileName_;
}

string Simulator::getStimulusFileName() const
{
   return stimulusFileName_;
}

Model &Simulator::getModel() const
{
   return *model_;
}

#ifdef PERFORMANCE_METRICS
Timer &Simulator::getTimer()
{
   return timer;
}

Timer &Simulator::getShort_timer()
{
   return short_timer;
}
#endif
///@}
