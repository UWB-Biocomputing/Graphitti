/**
 * @brief An interface for Neural Network Models.
 *
 * @class IModel IModel.h "IModel.h"
 *
 * Neural Network Model interface.
 *
 * Implementations define behavior of the network specific to the model. Specifically, a model
 * implementation handles:
 * * I/O
 * * Network creation
 * * Network simulation
 *
 * It is recommended that mutations of model state, if any, are avoided during a simulation. Some
 * models, such as those with complex connection dynamics or network history, may need to modify an
 * internal state during simulation.
 *
 * This is a pure interface and, thus, not directly instanced.
 *
 */

#pragma once
#include <iostream>

using namespace std;

#include "tinyxml.h"
#include "Global.h"
#include "Simulator.h"

#include "IAllNeurons.h"               // ToDo: Why are these so high up?
#include "IAllSynapses.h"              // ToDo: Why are these so high up?


// ToDo: get rid of all the methods, retain the class. only reexpose the methods actually necessary
//  as we find that they are necessary. Only expose methods that later on we find out we need.

class IModel {

public:

    /// Destructor
    virtual ~IModel() { }

    /************************************************
    *  Network IO Methods
    *  ToDo: should this be under Recording Methods?
    ***********************************************/

    /// Writes simulation results to an output destination.
    virtual void saveData() = 0;

    /************************************************
    *  Network Simulation Methods
    ***********************************************/

    /// Set up model state, if anym for a specific simulation run.
    virtual void setupSim() = 0;

    /// Advances network state one simulation step. ToDo: why is this not in Model.h?
    /// ToDo: this might be only thing left in model. with setup sim.
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
    /// behavior over the past epoch. Should be called once every epoch. ToDo: why is this not in Model.h?
    /// ToDo: Look at why simulator calls model->updateconnections
    /// might be similar to advance.
    virtual void updateConnections() = 0;

    /// Performs any finalization tasks on network following a simulation.
    virtual void cleanupSim() = 0;

   /************************************************
    *  Recording Methods
    *  ToDo: add recording methods to chain of responsibility.
    ***********************************************/

    /// Prints debug information about the current state of the network.
    virtual void logSimStep() const = 0;

    /// Copy GPU Synapse data to CPU.
    /// ToDo:
    virtual void copyGPUSynapseToCPUModel() = 0;

    /// Copy CPU Synapse data to GPU.
    virtual void copyCPUSynapseToGPUModel() = 0;

    /// Update the simulation history of every epoch.
    virtual void updateHistory() = 0;

   /************************************************
    *  Accessors
    *  ToDo: to get eliminated. Model wouldn't be owning these.
    ***********************************************/

    /// Get the IAllNeurons class object.
    /// @return Pointer to the AllNeurons class object.
    virtual IAllNeurons* getNeurons() = 0;

   // ToDo: Why no get Synapses (to be edges)?

    /// Get the Connections class object.
    /// @return Pointer to the Connections class object.
    virtual Connections* getConnections() = 0;

    /// Get the Layout class object.
    /// @return Pointer to the Layout class object.
    virtual Layout* getLayout() = 0;

};
