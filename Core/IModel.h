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
#include "IAllNeurons.h"               // ToDo: Why are these so high up?
#include "IAllSynapses.h"              // ToDo: Why are these so high up?
#include "Simulator.h"                 // ToDo: Why are these so high up?
#include "IRecorder.h"                 // ToDo: Why are these so high up?
#include "Connections.h"               // ToDo: Why are these so high up?
#include "Layout.h"                    // ToDo: Why are these so high up?

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
    virtual void advance() = 0;

    /// Modifies connections between neurons based on current state of the network and
    /// behavior over the past epoch. Should be called once every epoch. ToDo: why is this not in Model.h?
    /// ToDo: Look at why simulator calls model->updateconnections
    virtual void updateConnections() = 0;

    /// Performs any finalization tasks on network following a simulation.
    virtual void cleanupSim() = 0;

   /************************************************
    *  Recording Methods
    ***********************************************/

    /// Prints debug information about the current state of the network.
    virtual void logSimStep() const = 0;

    /// Copy GPU Synapse data to CPU.
    virtual void copyGPUSynapseToCPUModel() = 0;

    /// Copy CPU Synapse data to GPU.
    virtual void copyCPUSynapseToGPUModel() = 0;

    /// Update the simulation history of every epoch.
    virtual void updateHistory() = 0;

   /************************************************
    *  Accessors
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
