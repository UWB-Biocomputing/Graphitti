/**
 * @file Model
 *
 * @brief Implementation of Model for the spiking neural networks.
 *
 * The network is composed of 3 superimposed 2-d arrays: neurons, synapses, and
 * summation points.
 *
 * Synapses in the synapse map are located at the coordinates of the neuron
 * from which they receive output.  Each synapse stores a pointer into a
 * summation point. 
 *
 */



#pragma once

//#include "Coordinate.h"
//#include "Layouts/Layout.h"
//#include "SynapseIndexMap.h"
//#include "Simulator.h"
//#include "Global.h"
//#include "IAllNeurons.h"
//
//#include <vector>
//#include <iostream>
//#include <Simulation/Connections/Connections.h>

#include "Layout.h"
#include "IAllNeurons.h"

class Connections;

using namespace std;

class Model {
public:
   /// Constructor
   Model(
         /// factory class knows which synapse/neuron class to make.
         // ToDo: since these are getting created in factory fclassofcategory, these stay here
         Connections *conns,
         Layout *layout);

   /// Destructor
   virtual        ~Model();

   /// Writes simulation results to an output destination.
   /// Downstream from IModel saveData()
   // todo: put in chain of responsibility.
   virtual void saveData();

   /// Set up model state, for a specific simulation run.
   /// Downstream from IModel setupSim()
   virtual void setupSim();

   /// Performs any finalization tasks on network following a simulation.
   /// Downstream from IModel cleanupSim()
   virtual void cleanupSim();

   /// Update the simulation history of every epoch.
   virtual void updateHistory();

   /// Advances network state one simulation step.
   /// accessors (getNeurons, etc. owned by advance.)
   /// advance has detailed control over what does what when.
   /// detailed, low level control. clear onn what is happening when, how much time it is taking.
   /// If, during an advance cycle, a neuron \f$A\f$ at coordinates \f$x,y\f$ fires, every synapse
   /// which receives output is notified of the spike. Those synapses then hold
   /// the spike until their delay period is completed.  At a later advance cycle, once the delay
   /// period has been completed, the synapses apply their PSRs (Post-Synaptic-Response) to
   /// the summation points.
   /// Finally, on the next advance cycle, each neuron \f$B\f$ adds the value stored
   /// in their corresponding summation points to their \f$V_m\f$ and resets the summation points to zero.
   virtual void advance() = 0;

   /// Modifies connections between neurons based on current state of the network and
   /// behavior over the past epoch. Should be called once every epoch.
   /// ToDo: Look at why simulator calls model->updateconnections
   /// might be similar to advance.
   virtual void updateConnections() = 0;

   Connections *getConnections();

   Layout *getLayout();

protected:

   /// Prints debug information about the current state of the network.
   void logSimStep() const;

   /// error handling for read params
   // ToDo: do we need this?
   int read_params_;

   /// Copy GPU Synapse data to CPU.
   virtual void copyGPUtoCPU() = 0;

   /// Copy CPU Synapse data to GPU.
   virtual void copyCPUtoGPU() = 0;

protected:
   // DONE: 2020/03/14 (It was Emily!) Modified access level to public for allowing the access in BGDriver for serialization/deserialization
   // ToDo: make private again after serialization is fixed... shouldn't these be private with public accessors?
   // ToDo: Should model own these? Or should simulator?
   Connections *conns_;  // ToDo: make shared pointers

   // todo: have connections own synapses, have layouts own neurons

   Layout *layout_;

   // ToDo: Delete these and put them in connections and synapses accordingly
   IAllNeurons *neurons_;
   IAllSynapses *synapses_;

   // todo: put synapse index map in connections.
   // todo: how do synapses get neurons, neurons get synapses. should have member variable.
   SynapseIndexMap *synapseIndexMap_;

   void createAllNeurons(); /// Populate an instance of IAllNeurons with an initial state for each neuron.
};
