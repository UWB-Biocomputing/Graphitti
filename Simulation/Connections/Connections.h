/**
 * @file Connections.h
 * 
 * @ingroup Simulation/Connections
 *
 * @brief The base class of all connections classes
 *
 * @class Connections Connections.h "Connections.h"
 *
 * \latexonly  \subsubsection*{Implementation} \endlatexonly
 * \htmlonly   <h3>Implementation</h3> \endhtmlonly
 *
 * A placeholder to define connections of neunal networks.
 * In neunal networks, neurons are connected through synapses where messages are exchanged.
 * The strength of connections is characterized by synapse's weight. 
 * The connections classes define topologies, the way to connect neurons,  
 * and dynamics, the way to change connections as time elapses, of the networks. 
 * 
 * Connections can be either static or dynamic. The static connectons are ones where
 * connections are established at initialization and never change. 
 * The dynamic connections can be changed as the networks evolve, so in the dynamic networks
 * synapses will be created, deleted, or their weight will be modifed.  
 *
 * Connections classes may maintains intra-epoch state of connections in the network. 
 * This includes history and parameters that inform how new connections are made during growth.
 * Therefore, connections classes will have customized recorder classes, and provide
 * a function to craete the recorder class.
 *
 * \latexonly  \subsubsection*{Credits} \endlatexonly
 * \htmlonly   <h3>Credits</h3> \endhtmlonly
 *
 * Some models in this simulator is a rewrite of CSIM (2006) and other
 * work (Stiber and Kawasaki (2007?))
 */

#pragma once

#include <memory>

#include <log4cplus/loggingmacros.h>

#include "IAllNeurons.h"
#include "IAllSynapses.h"
#include "AllSpikingNeurons.h"
#include "AllSpikingSynapses.h"
#include "Layout.h"
#include "IRecorder.h"
#include "SynapseIndexMap.h"

using namespace std;

class Connections {
public:
   Connections();

   ///
   ///  Destructor
   ///
   virtual ~Connections();

   ///
   /// Returns shared pointer to Synapses/Edges 
   ///
   shared_ptr<IAllSynapses> getSynapses() const;


   ///
   /// Returns a shared pointer to the SynapseIndexMap
   ///
   shared_ptr<SynapseIndexMap> getSynapseIndexMap() const;

   ///
   /// Calls Synapses to create SynapseIndexMap and stores it as a member variable
   ///
   void createSynapseIndexMap();

   ///
   ///  Setup the internal structure of the class (allocate memories and initialize them).
   ///
   ///  @param  layout    Layout information of the neunal network.
   ///  @param  neurons   The Neuron list to search from.
   ///  @param  synapses  The Synapse list to search from.
   ///
   virtual void setupConnections(Layout *layout, IAllNeurons *neurons, IAllSynapses *synapses) = 0;

   ///
   /// Load member variables from configuration file.
   /// Registered to OperationManager as Operations::op::loadParameters
   ///
   virtual void loadParameters() = 0;

   ///
   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   ///
   virtual void printParameters() const = 0;

   ///
   ///  Update the connections status in every epoch.
   ///
   ///  @param  neurons  The Neuron list to search from.
   ///  @param  layout   Layout information of the neunal network.
   ///  @return true if successful, false otherwise.
   ///
   virtual bool updateConnections(IAllNeurons &neurons, Layout *layout);

   ///
   ///  Creates synapses from synapse weights saved in the serialization file.
   ///
   ///  @param  numNeurons Number of neurons to update.
   ///  @param  layout      Layout information of the neunal network.
   ///  @param  ineurons    The Neuron list to search from.
   ///  @param  isynapses   The Synapse list to search from.
   ///
   void
   createSynapsesFromWeights(const int numNeurons, Layout *layout, IAllNeurons &ineurons, IAllSynapses &isynapses);

#if defined(USE_GPU)
   public:
       ///
       ///  Update the weight of the Synapses in the simulation.
       ///  Note: Platform Dependent.
       ///
       ///  @param  numNeurons          number of neurons to update.
       ///  @param  neurons             the Neuron list to search from.
       ///  @param  synapses            the Synapse list to search from.
       ///  @param  allNeuronsDevice    GPU address of the allNeurons struct on device memory.
       ///  @param  allSynapsesDevice   GPU address of the allSynapses struct on device memory.
       ///  @param  layout              Layout information of the neunal network.
       ///
       virtual void updateSynapsesWeights(const int numNeurons, IAllNeurons &neurons, IAllSynapses &synapses, AllSpikingNeuronsDeviceProperties* allNeuronsDevice, AllSpikingSynapsesDeviceProperties* allSynapsesDevice, Layout *layout);
#else
public:
   ///
   ///  Update the weight of the Synapses in the simulation.
   ///  Note: Platform Dependent.
   ///
   ///  @param  numNeurons Number of neurons to update.
   ///  @param  ineurons    The Neuron list to search from.
   ///  @param  isynapses   The Synapse list to search from.
   ///
   virtual void
   updateSynapsesWeights(const int numNeurons, IAllNeurons &neurons, IAllSynapses &synapses, Layout *layout);

#endif // USE_GPU

protected:

   shared_ptr<IAllSynapses> synapses_;

   shared_ptr<SynapseIndexMap> synapseIndexMap_;

   log4cplus::Logger fileLogger_;
};

