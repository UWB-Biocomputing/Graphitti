/**
 * @file Layout.h
 * 
 * @ingroup Simulator/Layouts
 *
 * @brief The Layout class defines the layout of neurons in neural networks
 * Implementation:
 * The Layout class maintains neurons locations (x, y coordinates),
 * distance of every couple neurons,
 * neurons type map (distribution of excitatory and inhibitory neurons),
 * and starter neurons map
 * (distribution of endogenously active neurons).  
 */

#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include <log4cplus/loggingmacros.h>

#include "Utils/Global.h"
#include "IAllVertices.h"

using namespace std;

class Layout {
public:
   Layout();

   virtual ~Layout();

   shared_ptr<IAllVertices> getVertices() const;

   /// Setup the internal structure of the class.
   /// Allocate memories to store all layout state.
   virtual void setupLayout();

   /// Load member variables from configuration file. Registered to OperationManager as Operation::loadParameters
   virtual void loadParameters();

   /// Prints out all parameters to logging file. Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const;

   /// Creates a neurons type map.
   /// @param  numNeurons number of the neurons to have in the type map.
   virtual void generateNeuronTypeMap(int numNeurons);

   /// Populates the starter map.
   /// Selects num_endogenously_active_neurons excitory neurons
   /// and converts them into starter neurons.
   /// @param  numNeurons number of neurons to have in the map.
   virtual void initStarterMap(const int numNeurons);


   /// Returns the type of synapse at the given coordinates
   /// @param    srcNeuron  integer that points to a Neuron in the type map as a source.
   /// @param    destNeuron integer that points to a Neuron in the type map as a destination.
   /// @return type of the synapse.
   synapseType synType(const int srcNeuron, const int destNeuron);

   VectorMatrix *xloc_;  ///< Store neuron i's x location.

   VectorMatrix *yloc_;   ///< Store neuron i's y location.

   CompleteMatrix *dist2_;  ///< Inter-neuron distance squared.

   CompleteMatrix *dist_;    ///< The true inter-neuron distance.

   vector<int> probedNeuronList_;   ///< Probed neurons list. // ToDo: Move this to Hdf5 recorder once its implemented in project -chris

   neuronType *neuronTypeMap_;    ///< The neuron type map (INH, EXC).

   bool *starterMap_; ///< The starter existence map (T/F).

   BGSIZE numEndogenouslyActiveNeurons_;    ///< Number of endogenously active neurons.

protected:
   shared_ptr<IAllVertices> neurons_;

   vector<int> endogenouslyActiveNeuronList_;    ///< Endogenously active neurons list.

   vector<int> inhibitoryNeuronLayout_;    ///< Inhibitory neurons list.

   log4cplus::Logger fileLogger_;

private:
   /// initialize the location maps (xloc and yloc).
   void initNeuronsLocs();

   bool gridLayout_;    ///< True if grid layout.

};

