/**
 *      @file FixedLayout.h
 *
 *      @brief The Layout class defines the layout of neurons in neunal networks
 */

/**
 *
 * @class FixedLayout FixedLayout.h "FixedLayout.h"
 *
 * \latexonly  \subsubsection*{Implementation} \endlatexonly
 * \htmlonly   <h3>Implementation</h3> \endhtmlonly
 *
 * The FixedLayout class maintains neurons locations (x, y coordinates), 
 * distance of every couple neurons,
 * neurons type map (distribution of excitatory and inhibitory neurons), and starter neurons map
 * (distribution of endogenously active neurons).  
 *
 * The FixedLayout class reads all layout information from parameter description file.
 *
 */

#pragma once

#include "Layout.h"

using namespace std;

class FixedLayout : public Layout {
public:
   FixedLayout();

   virtual ~FixedLayout();

   static Layout *Create() { return new FixedLayout(); }

   /**
    *  Prints out all parameters to logging file.
    *  Registered to OperationManager as Operation::printParameters
    */
   virtual void printParameters() const;

   /**
    *  Creates a neurons type map.
    *
    *  @param  numNeurons number of the neurons to have in the type map.
    */
   virtual void generateNeuronTypeMap(int numNeurons);

   /**
    *  Populates the starter map.
    *  Selects num_endogenously_active_neurons excitory neurons
    *  and converts them into starter neurons.
    *
    *  @param  numNeurons number of neurons to have in the map.
    */
   virtual void initStarterMap(const int numNeurons);
};

