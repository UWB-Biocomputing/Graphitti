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

#include <memory>

#include "BGTypes.h"
#include "Global.h"
#include "Core/Model.h"
//#include "GPUSpikingModel.h"
//#include "CPUSpikingModel.h"
//#include "ISInput.h"
#include "Timer.h"

class Model;

#ifdef PERFORMANCE_METRICS
// Home-brewed performance measurement  *doesnt affect runtime itself. *also collects performance on GPU *warner smidt paper with details "profiling braingrid"
#include "Timer.h"
#endif

// ToDo: siminfo had TiXmlVisitor. Wondering if still needed?
class Simulator {
public:
   static Simulator &getInstance(); /// Acts as constructor, returns the instance of singleton object
   virtual ~Simulator(); /// Destructor

   void setup(); /// Setup simulation.

   void finish(); /// Cleanup after simulation.

   void loadParameters(); /// Load member variables from configuration file

   void printParameters() const; /// Prints out loaded parameters to console.

   void copyGPUSynapseToCPU(); /// Copy GPU Synapse data to CPU.

   void copyCPUSynapseToGPU(); /// Copy CPU Synapse data to GPU.

   void reset(); /// Reset simulation objects.

   void simulate();

   void advanceUntilGrowth(
         const int &currentEpoch) const; /// Advance simulation to next growth cycle. Helper for #simulate().

   void saveData() const; /// Writes simulation results to an output destination.

   /// Instantiates Model which causes all other lower level simulator objects to be instantiated. Checks if all
   /// expected objects were created correctly and returns T/F on the success of the check.
   bool instantiateSimulatorObjects();

/************************************************
 *  Accessors
 ***********************************************/
   int getWidth() const;   /// Width of neuron map (assumes square)

   int getHeight() const;  /// Height of neuron map

   int getTotalNeurons() const;  /// Count of neurons in the simulation

   int getCurrentStep() const;    /// Current simulation step

   int getNumEpochs() const;   /// Maximum number of simulation steps

   BGFLOAT getEpochDuration() const;    /// The length of each step in simulation time

   int getMaxFiringRate() const;   /// Maximum firing rate. **GPU Only**

   int getMaxSynapsesPerNeuron() const;   /// Maximum number of synapses per neuron. **GPU Only**

   BGFLOAT getDeltaT() const;    /// Time elapsed between the beginning and end of the simulation step

   neuronType *getRgNeuronTypeMap() const;    /// The neuron type map (INH, EXC).

   bool *getRgEndogenouslyActiveNeuronMap() const;  /// The starter existence map (T/F).

   BGFLOAT getMaxRate() const;   /// growth variable (m_targetRate / m_epsilon) TODO: more detail here

   BGFLOAT *getPSummationMap() const;   /// List of summation points (either host or device memory)

   long getSeed() const;    /// Seed used for the simulation random **SingleThreaded Only**

   string getResultFileName() const;    /// File name of the simulation results.

   string getConfigFileName() const;    /// File name of the parameter configuration file.

   string getSerializationFileName() const;    /// File name of the serialization file.

   string getDeserializationFileName() const; /// File name of the deserialization file.

   string getStimulusFileName() const;     /// File name of the stimulus input file.

   shared_ptr<Model> getModel() const;    /// Neural Network Model interface.

/************************************************
 *  Mutators
 ***********************************************/
   void setPSummationMap(BGFLOAT *summationMap);     /// Mutator for summation map (added late)

   void setResultFileName(const string &fileName);

   void setConfigFileName(const string &fileName);

   void setSerializationFileName(const string &fileName);

   void setDeserializationFileName(const string &fileName);

   void setStimulusFileName(const string &fileName);

#ifdef PERFORMANCE_METRICS
   Timer getTimer();  /// Timer measures performance of epoch. returns copy of internal timer owned by simulator.
   Timer getShort_timer(); ///Timer for measuring performance of connection update.
#endif

   /// Delete these methods because they can cause copy instances of the singleton when using threads.
   Simulator(Simulator const &) = delete;

   void operator=(Simulator const &) = delete;

private:
   Simulator();    /// Constructor is private to keep a singleton instance of this class.

   void freeResources(); /// Frees dynamically allocated memory associated with the maps.

   int width_; /// Width of neuron map (assumes square)

   int height_;   /// Height of neuron map

   int totalNeurons_;   /// Count of neurons in the simulation

   int currentEpoch_;   /// Current epoch step

   int numEpochs_; /// Number of simulator epochs

   BGFLOAT epochDuration_; /// The length of each step in simulation time

   int maxFiringRate_;  /// Maximum firing rate. **GPU Only**

   int maxSynapsesPerNeuron_;  /// Maximum number of synapses per neuron. **GPU Only**

   BGFLOAT deltaT_;   /// Inner Simulation Step Duration, purely investigative.

   neuronType *rgNeuronTypeMap_; /// The neuron type map (INH, EXC). ToDo: become a vector

   bool *rgEndogenouslyActiveNeuronMap_;   /// The starter existence map (T/F). ToDo: become a vector

   BGFLOAT maxRate_;   /// growth variable (m_targetRate / m_epsilon) TODO: more detail here

   BGFLOAT *pSummationMap_;    /// List of summation points (either host or device memory) ToDo: make smart ptr

   long seed_;   /// Seed used for the simulation random SINGLE THREADED

   string resultFileName_;    /// File name of the simulation results.

   string configFileName_;    /// File name of the parameter configuration file.

   string serializationFileName_;    /// File name of the serialization file.

   string deserializationFileName_;    /// File name of the deserialization file.

   string stimulusFileName_;    /// File name of the stimulus input file.

   shared_ptr<Model> model_;  /// Smart pointer to model class (Model is an interface class)

#ifdef PERFORMANCE_METRICS
   Timer timer;   /// Timer for measuring performance of an epoch.
   Timer short_timer; /// Timer for measuring performance of connection update.
#endif
};



 