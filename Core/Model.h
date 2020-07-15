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
 * The Model class maintains and manages classes of objects that make up
 * essential components of the spiking neunal network.
 *    -# IAllNeurons: A class to define a list of partiular type of neurons.
 *    -# IAllSynapses: A class to define a list of partiular type of synapses.
 *    -# Connections: A class to define connections of the neunal network.
 *    -# Layout: A class to define neurons' layout information in the network.
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
 * If, during an advance cycle, a neuron \f$A\f$ at coordinates \f$x,y\f$ fires, every synapse
 * which receives output is notified of the spike. Those synapses then hold
 * the spike until their delay period is completed.  At a later advance cycle, once the delay
 * period has been completed, the synapses apply their PSRs (Post-Synaptic-Response) to 
 * the summation points.  
 * Finally, on the next advance cycle, each neuron \f$B\f$ adds the value stored
 * in their corresponding summation points to their \f$V_m\f$ and resets the summation points to
 * zero.
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
          Connections *conns,
          IAllNeurons *neurons,
          IAllSynapses *synapses,
          Layout *layout);

   /// Destructor
   virtual        ~Model();

   /// Writes simulation results to an output destination.
   /// Downstream from IModel saveData()
   virtual void   saveData();

   /// Set up model state, for a specific simulation run.
   /// Downstream from IModel setupSim()
   virtual void   setupSim();

   /// Performs any finalization tasks on network following a simulation.
   /// Downstream from IModel cleanupSim()
   virtual void   cleanupSim();

   /// returns ptr for AllNeurons class object.
   virtual        IAllNeurons* getNeurons();

   /// returns ptr for Connections class object.
   virtual        Connections* getConnections();

   /// returns ptr for Layouts class object.
   virtual        Layout* getLayout();

   /// Update the simulation history of every epoch.
   virtual void   updateHistory();

   /// Copy GPU Synapse data to CPU.
   virtual void   copyGPUSynapseToCPUModel() = 0;

   ///  Copy CPU Synapse data to GPU.
   virtual void   copyCPUSynapseToGPUModel() = 0;

protected:

   /// Prints debug information about the current state of the network.
   void logSimStep() const;

   /// error handling for read params
   // ToDo: do we need this?
   int m_read_params;

public:
   // 2020/03/14 Modified access level to public for allowing the access in BGDriver for serialization/deserialization
   // ToDo: make private again after serialization is fixed... shouldn't these be private with public accessors?
   // ToDo: Should model own these? Or should simulator?
   Connections    *m_conns;

   IAllNeurons    *m_neurons;

   IAllSynapses   *m_synapses;

   Layout         *m_layout;

   SynapseIndexMap *m_synapseIndexMap;

private:
        void createAllNeurons(); /// Populate an instance of IAllNeurons with an initial state for each neuron.

        std::weak_ptr<Simulator>() simulator;  /// Weak ptr to instance of simulator

};
