/**
 * @file Model.h
 *
 * @brief Implementation of Model for the spiking neunal networks.
 *
 * @class Model Model.h "Model.h"
 *
 * \latexonly  \subsubsection*{Implementation} \endlatexonly
 * \htmlonly   <h3>Implementation</h3> \endhtmlonly
 *
 * \image html bg_data_layout.png
 *
 * The network is composed of 3 superimposed 2-d arrays: neurons, synapses, and
 * summation points.
 *
 * Synapses in the synapse map are located at the coordinates of the neuron
 * from which they receive output.  Each synapse stores a pointer into a
 * summation point. 
 *
 *
 * \latexonly  \subsubsection*{Credits} \endlatexonly
 * \htmlonly   <h3>Credits</h3> \endhtmlonly
 */

#pragma once

#include "IModel.h"
#include "Coordinate.h"
#include "Layout.h"
#include "SynapseIndexMap.h"
#include "Simulator.h"

#include <vector>
#include <iostream>

using namespace std;

class Model : public IModel // ToDo: is this supposed to be protected?
{

public:

    /// Constructor
    Model(
          /// factory class knows which synapse/neuron class to make.
          // ToDo: since these are getting created in factory fclassofcategory, these stay here
          Connections *conns,
          IAllNeurons *neurons,
          IAllSynapses *synapses,
          Layout *layout);

   /// Destructor
   virtual        ~Model();

   /// Writes simulation results to an output destination.
   /// Downstream from IModel saveData()
   // todo: put in chain of responsibility.
   virtual void   saveData();

   /// Set up model state, for a specific simulation run.
   /// Downstream from IModel setupSim()
   virtual void   setupSim();

   /// Performs any finalization tasks on network following a simulation.
   /// Downstream from IModel cleanupSim()
   virtual void   cleanupSim();

   /// Update the simulation history of every epoch.
   virtual void   updateHistory();

   /// todo: where is advance? is it in gpu cpu ?

   // todo: advance and update connections.

protected:

   /// Prints debug information about the current state of the network.
   void logSimStep() const;

   /// error handling for read params
   // ToDo: do we need this?
   int m_read_params;

private:
   // DONE: 2020/03/14 (It was Emily!) Modified access level to public for allowing the access in BGDriver for serialization/deserialization
   // ToDo: make private again after serialization is fixed... shouldn't these be private with public accessors?
   // ToDo: Should model own these? Or should simulator?
   Connections    *m_conns;  // ToDo: make shared pointers

   // todo: have connections own synapses, have layouts own neurons

   Layout         *m_layout;

   // todo: put synapse index map in connections.
   // todo: how do synapses get neurons, neurons get synapses. should have member variable.
   SynapseIndexMap *m_synapseIndexMap;


        void createAllNeurons(); /// Populate an instance of IAllNeurons with an initial state for each neuron.

         // todo: get rid of ptr. sim should return &reference in getinsstance
        std::weak_ptr<Simulator>() simulator;  /// Weak ptr to instance of simulator

};
