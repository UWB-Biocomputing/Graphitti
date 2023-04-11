/**
 * @file Simulator.h
 *
 * @brief Platform independent base class for the Brain Grid simulator.
 * Simulator is a singleton class (a class that can only have one object)
 *
 * @ingroup Simulator/Core
 */

#pragma once

// ToDo: revisit big decisions here for high level architecture

#include "BGTypes.h"
#include "Core/Model.h"
#include "Global.h"
#include <log4cplus/loggingmacros.h>
#include <memory>
//#include "GPUModel.h"
//#include "CPUModel.h"
//#include "ISInput.h"
#include "Timer.h"

class Model;

class Simulator {
public:
   static Simulator &getInstance();   /// Acts as constructor, returns the instance
                                      /// of singleton object
   virtual ~Simulator() = default;    /// Destructor

   void setup();   /// Setup simulation.

   void finish();   /// Cleanup after simulation.

   void loadParameters();   /// Load member variables from configuration file

   void printParameters() const;   /// Prints loaded parameters to logging file.

   // Copied over from STDPFix
   void copyGPUSynapseToCPU();   /// Copy GPU Synapse data to CPU.

   // Copied over from STDPFix
   void copyCPUSynapseToGPU();   /// Copy CPU Synapse data to GPU.

   void reset();   /// Reset simulation objects.

   void simulate();

   void advanceEpoch(const int &currentEpoch) const;   /// Advance simulation to next growth
                                                       /// cycle. Helper for #simulate().

   void saveResults() const;   /// Writes simulation results to an output destination.

   /// Instantiates Model which causes all other lower level simulator objects to
   /// be instantiated. Checks if all expected objects were created correctly and
   /// returns T/F on the success of the check.
   bool instantiateSimulatorObjects();

   /************************************************
   *  Accessors
   ***********************************************/
   ///@{
   int getWidth() const;   /// Width of neuron map (assumes square)

   int getHeight() const;   /// Height of neuron map

   int getTotalVertices() const;   /// Count of neurons in the simulation

   int getCurrentStep() const;   /// Current simulation step

   int getNumEpochs() const;   /// Maximum number of simulation steps

   BGFLOAT
   getEpochDuration() const;   /// The length of each step in simulation time

   int getMaxFiringRate() const;   /// Maximum firing rate. **GPU Only**

   int getMaxEdgesPerVertex() const;   /// Maximum number of synapses per neuron. **GPU Only**

   BGFLOAT
   getDeltaT() const;   /// Time elapsed between the beginning and end of the
                        /// simulation step

   BGFLOAT
   getMaxRate() const;   /// growth variable (m_targetRate / m_epsilon) TODO: more
                         /// detail here

   long getNoiseRngSeed() const;   /// Seed used for the simulation random **SingleThreaded Only**

   long getInitRngSeed() const;   /// Seed used to initialize parameters

   string getConfigFileName() const;   /// File name of the parameter configuration file.

   string getSerializationFileName() const;   /// File name of the serialization file.

   string getDeserializationFileName() const;   /// File name of the deserialization file.

   string getStimulusFileName() const;   /// File name of the stimulus input file.

   Model *getModel() const;   /// Neural Network Model interface.
                              ///@}

   /************************************************
   *  Mutators
   ***********************************************/
   ///@{
   void setConfigFileName(const string &fileName);

   void setSerializationFileName(const string &fileName);

   void setDeserializationFileName(const string &fileName);

   void setStimulusFileName(const string &fileName);

#ifdef PERFORMANCE_METRICS
   Timer &getTimer();         /// Timer measures performance of epoch. returns copy of
                              /// internal timer owned by simulator.
   Timer &getShort_timer();   /// Timer for measuring performance of connection update.
#endif

   /// Delete copy and move methods to avoid copy instances of the singleton
   Simulator(const Simulator &simulator) = delete;
   Simulator &operator=(const Simulator &simulator) = delete;

   Simulator(Simulator &&simulator) = delete;
   Simulator &operator=(Simulator &&simulator) = delete;

private:
   Simulator();   /// Constructor is private to keep a singleton instance of this
                  /// class.

   int width_;   /// Width of neuron map (assumes square)

   int height_;   /// Height of neuron map

   int totalNeurons_;   /// Count of neurons in the simulation

   int currentEpoch_;   /// Current epoch step

   int numEpochs_;   /// Number of simulator epochs

   BGFLOAT epochDuration_;   /// The length of each epoch in simulation time

   int maxFiringRate_;   /// Maximum firing rate. **GPU Only**

   int maxEdgesPerVertex_;   /// Maximum number of synapses per neuron. **GPU
                             /// Only**

   BGFLOAT deltaT_;   /// Inner Simulation Step Duration, purely investigative.

   BGFLOAT maxRate_;   /// growth variable (m_targetRate / m_epsilon) TODO: more
                       /// detail here

   long noiseRngSeed_;   /// Seed used for the simulation random SINGLE THREADED

   long initRngSeed_;   /// Seed used to initialize parameters

   string configFileName_;   /// File name of the parameter configuration file.

   string serializationFileName_;   /// File name of the serialization file.

   string deserializationFileName_;   /// File name of the deserialization file.

   string stimulusFileName_;   /// File name of the stimulus input file.

   unique_ptr<Model> model_;   /// Smart pointer to model class (Model is an interface class)

   log4cplus::Logger
      consoleLogger_;   /// Logger for printing to the console as well as the logging file
   log4cplus::Logger fileLogger_;        /// Logger for printing to the logging file
   log4cplus::Logger edgeLogger_;        /// Logger for printing to the logging file
   log4cplus::Logger workbenchLogger_;   /// Logger for printing to logging file

#ifdef PERFORMANCE_METRICS
   Timer timer;         /// Timer for measuring performance of an epoch.
   Timer short_timer;   /// Timer for measuring performance of connection update.
#endif
   ///@}
};
