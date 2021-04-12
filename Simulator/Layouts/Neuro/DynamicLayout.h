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
   virtual void printParameters() const;

   ///  Creates a randomly ordered distribution with the specified numbers of neuron types.
   ///
   ///  @param  numVertices number of the neurons to have in the type map.
   virtual void generateVertexTypeMap(int numVertices);

   ///  Populates the starter map.
   ///  Selects num_endogenously_active_neurons excitory neurons
   ///  and converts them into starter neurons.
   ///
   ///  @param  numVertices number of vertices to have in the map.
   virtual void initStarterMap(const int numVertices);

   /// Load member variables from configuration file. Registered to OperationManager as Operation::loadParameters
   virtual void loadParameters(); 

private:
   /// Fraction of endogenously active neurons.
   BGFLOAT m_frac_starter_neurons;

   /// Fraction of exitatory neurons.
   BGFLOAT m_frac_excitatory_neurons;
};

