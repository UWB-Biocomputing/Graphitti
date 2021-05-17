/**
 * @file DynamicLayout.h
 * 
 * @ingroup Simulator/Layouts
 *
 * @brief The DynamicLayout class defines the layout of neurons in neural networks
 *
 * The DynamicLayout class maintains neurons locations (x, y coordinates), 
 * distance of every couple neurons,
 * neurons type map (distribution of excitatory and inhibitory neurons), and starter neurons map
 * (distribution of endogenously active neurons).  
 *
 * The DynamicLayout class generates layout information dynamically.
 */

#pragma once

#include "Layout.h"

using namespace std;

class DynamicLayout : public Layout {
public:
   DynamicLayout();

   virtual ~DynamicLayout();

   static Layout *Create() { return new DynamicLayout(); }

   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const override;

   ///  Creates a randomly ordered distribution with the specified numbers of neuron types.
   ///
   ///  @param  numVertices number of the neurons to have in the type map.
   virtual void generateVertexTypeMap(int numVertices) override;

   ///  Populates the starter map.
   ///  Selects num_endogenously_active_neurons excitory neurons
   ///  and converts them into starter neurons.
   ///
   ///  @param  numVertices number of vertices to have in the map.
   virtual void initStarterMap(const int numVertices) override;

   /// Load member variables from configuration file. Registered to OperationManager as Operation::loadParameters
   virtual void loadParameters(); 

   /// Returns the type of synapse at the given coordinates
   /// @param    srcVertex  integer that points to a Neuron in the type map as a source.
   /// @param    destVertex integer that points to a Neuron in the type map as a destination.
   /// @return type of the synapse.
   virtual edgeType edgType(const int srcVertex, const int destVertex);

private:
   /// Fraction of endogenously active neurons.
   BGFLOAT fractionEndogenouslyActive_;

   /// Fraction of exitatory neurons.
   BGFLOAT fractionExcitatory_;
};

