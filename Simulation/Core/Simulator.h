/**
 * @file Simulator.h
 *
 * @brief Platform independent base class for the Brain Grid simulator.
 * Simulator is a singleton class (a class that can only have one object)
 *
 * @ingroup Core
 */

#pragma once

// ToDo: revisit big decisions here for high level architecture

#include "BGTypes.h"
#include "Global.h"
#include "Core/Model.h"
//#include "ISInput.h"
#include "Timer.h"

class IRecorder;
class Model;

#ifdef PERFORMANCE_METRICS
// Home-brewed performance measurement  *doesnt affect runtime itself. *also collects performance on GPU *warner smidt paper with details "profiling braingrid"
#include "Timer.h"
#endif

// ToDo: siminfo had TiXmlVisitor. Wondering if still needed?
class Simulator {
public:

   static Simulator &getInstance(); /// Acts as constructor, returns the instance of singleton object

   // ToDo: one accessor get current timestep
   // ToDo: make all getters const.

   int getWidth() const;   /// Width of neuron map (assumes square)

   int getHeight() const;  /// Height of neuron map

   int getTotalNeurons() const;  /// Count of neurons in the simulation

   int getCurrentStep() const;    /// Current simulation step

   int getMaxSteps() const;   /// Maximum number of simulation steps

   BGFLOAT getEpochDuration() const;    /// The length of each step in simulation time

   int getMaxFiringRate() const;   /// Maximum firing rate. **GPU Only**

   int getMaxSynapsesPerNeuron() const;   /// Maximum number of synapses per neuron. **GPU Only**

   BGFLOAT getDeltaT() const;    /// Time elapsed between the beginning and end of the simulation step

   neuronType *getRgNeuronTypeMap() const;    /// The neuron type map (INH, EXC).

   bool *getRgEndogenouslyActiveNeuronMap() const;  /// The starter existence map (T/F).

   BGFLOAT getMaxRate() const;   /// growth variable (m_targetRate / m_epsilon) TODO: more detail here

   BGFLOAT *getPSummationMap() const;   /// List of summation points (either host or device memory)

   void setPSummationMap(BGFLOAT *summationMap);     /// Mutator for summation map (added late)

   void setSimRecorder(IRecorder *recorder);

   long getSeed() const;    /// Seed used for the simulation random **SingleThreaded Only**

   string getStateOutputFileName() const;    /// File name of the simulation results.

   string getStateInputFileName() const;    /// File name of the parameter description file.

   string getMemOutputFileName() const;    /// File name of the memory dump output file.

   string getMemInputFileName() const; /// File name of the memory dump input file.

   string getStimulusInputFileName() const;     /// File name of the stimulus input file.

   Model *getModel() const;    /// Neural Network Model interface. ToDo: make smart ptr

   IRecorder *getSimRecorder() const;    /// Recorder object. ToDo: make smart ptr

   // ToDo: Questions
   // ToDo:  do we need to return these?
   // ToDo: do these need to be accessible outside?
   // ToDo: figure out something else to return beside ptr. maybe safely return const reference?
   // ISInput *getPInput();  /// Stimulus input object.

#ifdef PERFORMANCE_METRICS
   Timer getTimer();  /// Timer measures performance of epoch. returns copy of internal timer owned by simulator.
   Timer getShort_timer(); ///Timer for measuring performance of connection update.
#endif

   virtual ~Simulator(); /// Destructor

   void setup(); /// Setup simulation.

   void finish(); /// Cleanup after simulation.

   void printParameters(ostream &output) const; /// Prints out loaded parameters to ostream.

   void copyGPUSynapseToCPU(); /// Copy GPU Synapse data to CPU.

   void copyCPUSynapseToGPU(); /// Copy CPU Synapse data to GPU.

   void reset(); /// Reset simulation objects.

   void simulate();

   void advanceUntilGrowth(
           const int &currentStep) const; /// Advance simulation to next growth cycle. Helper for #simulate().

   void saveData() const; /// Writes simulation results to an output destination.

   /// Delete these methods because they can cause copy instances of the singleton when using threads.
   Simulator(Simulator const &) = delete;
   void operator=(Simulator const &) = delete;

private:

   Simulator(); /// Constructor

   void freeResources(); /// Frees dynamically allocated memory associated with the maps.

   int width; /// Width of neuron map (assumes square)

   int height;   /// Height of neuron map

   int totalNeurons;   /// Count of neurons in the simulation

   int currentStep;   /// Current simulation step

   int maxSteps; // TODO: delete /// Maximum number of simulation steps

   BGFLOAT epochDuration; /// The length of each step in simulation time

   int maxFiringRate;  /// Maximum firing rate. **GPU Only**

   int maxSynapsesPerNeuron;  /// Maximum number of synapses per neuron. **GPU Only**

   BGFLOAT deltaT;   /// Inner Simulation Step Duration, purely investigative.

   neuronType *rgNeuronTypeMap; /// The neuron type map (INH, EXC). ToDo: make smart ptr

   bool *rgEndogenouslyActiveNeuronMap;   /// The starter existence map (T/F). ToDo: make smart ptr

   BGFLOAT maxRate;   /// growth variable (m_targetRate / m_epsilon) TODO: more detail here

   BGFLOAT *pSummationMap;    /// List of summation points (either host or device memory) ToDo: make smart ptr

   long seed;   /// Seed used for the simulation random SINGLE THREADED

   string stateOutputFileName;    /// File name of the simulation results.

   string stateInputFileName;    /// File name of the parameter description file.

   string memOutputFileName;    /// File name of the memory dump output file.

   string memInputFileName;    /// File name of the memory dump input file.

   string stimulusInputFileName;    /// File name of the stimulus input file.

   Model *model;    /// Neural Network Model interface. ToDo: make smart ptr

   IRecorder *simRecorder;    /// ptr to Recorder object. ToDo: make smart ptr

   // ISInput *pInput;    /// Stimulus input object. ToDo: make smart ptr

#ifdef PERFORMANCE_METRICS
   Timer timer;   /// Timer for measuring performance of an epoch.
   Timer short_timer; /// Timer for measuring performance of connection update.
#endif
};



 