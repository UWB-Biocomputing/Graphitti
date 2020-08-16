/**
 *      @file Layout.h
 *
 *      @brief The Layout class defines the layout of neurons in neural networks
 */

/**
 *
 * @class Layout Layout.h "Layout.h"
 *
 *
 * Implementation:
 * The Layout class maintains neurons locations (x, y coordinates),
 * distance of every couple neurons,
 * neurons type map (distribution of excitatory and inhibitory neurons),
 * and starter neurons map
 * (distribution of endogenously active neurons).  
 *
 */

#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "Utils/Global.h"
#include "IAllNeurons.h"

using namespace std;

class Layout {
public:
   Layout();

   virtual ~Layout();

   shared_ptr<IAllNeurons> getNeurons() const;

   /// Setup the internal structure of the class.
   /// Allocate memories to store all layout state.
   virtual void setupLayout();

   /// Load member variables from configuration file. Registered to OperationManager as Operation::op::loadParameters
   virtual void loadParameters();

   /// Prints out all parameters of the neurons to ostream.
   /// @param  output  ostream to send output to.
   virtual void printParameters() const;

   /// Creates a neurons type map.
   /// @param  num_neurons number of the neurons to have in the type map.
   virtual void generateNeuronTypeMap(int num_neurons);

   /// Populates the starter map.
   /// Selects num_endogenously_active_neurons excitory neurons
   /// and converts them into starter neurons.
   /// @param  num_neurons number of neurons to have in the map.
   virtual void initStarterMap(const int num_neurons);


   /// Returns the type of synapse at the given coordinates
   /// @param    src_neuron  integer that points to a Neuron in the type map as a source.
   /// @param    dest_neuron integer that points to a Neuron in the type map as a destination.
   /// @return type of the synapse.
   synapseType synType(const int src_neuron, const int dest_neuron);

   VectorMatrix *xloc_;  ///< Store neuron i's x location.

   VectorMatrix *yloc_;   ///< Store neuron i's y location.

   CompleteMatrix *dist2_;  ///< Inter-neuron distance squared.

   CompleteMatrix *dist_;    ///< The true inter-neuron distance.

   vector<int> probedNeuronList_;   ///< Probed neurons list. // ToDo: Move this to Hdf5 recorder once its implemented in project

   neuronType *neuronTypeMap_;    ///< The neuron type map (INH, EXC).

   bool *starterMap_; ///< The starter existence map (T/F).

   BGSIZE numEndogenouslyActiveNeurons_;    ///< Number of endogenously active neurons.

protected:
   shared_ptr<IAllNeurons> neurons_;

   vector<int> endogenouslyActiveNeuronList_;    ///< Endogenously active neurons list.

   vector<int> inhibitoryNeuronLayout_;    ///< Inhibitory neurons list.

private:
   /// initialize the location maps (xloc and yloc).
   void initNeuronsLocs();

   bool gridLayout_;    ///< True if grid layout.

};

