/**
 * @file FixedLayout.cpp
 *
 * @ingroup Simulator/Layouts
 * 
 * @brief 
 */

#include "FixedLayout.h"
#include "ParseParamError.h"
#include "Util.h"

FixedLayout::FixedLayout() : Layout() {
}

FixedLayout::~FixedLayout() {
}

///  Prints out all parameters to logging file.
///  Registered to OperationManager as Operation::printParameters
void FixedLayout::printParameters() const {
   Layout::printParameters();

   LOG4CPLUS_DEBUG(fileLogger_, "\n\tLayout type: FixedLayout" << endl << endl);
}

///  Creates a randomly ordered distribution with the specified numbers of neuron types.
///
///  @param  numNeurons number of the neurons to have in the type map.
void FixedLayout::generateNeuronTypeMap(int numNeurons) {
   Layout::generateNeuronTypeMap(numNeurons);

   int numInhibitoryNeurons = inhibitoryNeuronLayout_.size();
   int numExcititoryNeurons = numNeurons - numInhibitoryNeurons;

   LOG4CPLUS_DEBUG(fileLogger_, "\nNEURON TYPE MAP" << endl
   << "\tTotal neurons: " << numNeurons << endl
   << "\tInhibitory Neurons: " << numInhibitoryNeurons << endl
   << "\tExcitatory Neurons: " << numExcititoryNeurons << endl);

   for (int i = 0; i < numInhibitoryNeurons; i++) {
      assert(inhibitoryNeuronLayout_.at(i) < numNeurons);
      neuronTypeMap_[inhibitoryNeuronLayout_.at(i)] = INH;
   }

   LOG4CPLUS_INFO(fileLogger_, "Finished initializing neuron type map");
}

///  Populates the starter map.
///  Selects \e numStarter excitory neurons and converts them into starter neurons.
///  @param  numNeurons number of neurons to have in the map.
void FixedLayout::initStarterMap(const int numNeurons) {
   Layout::initStarterMap(numNeurons);

   for (BGSIZE i = 0; i < numEndogenouslyActiveNeurons_; i++) {
      assert(endogenouslyActiveNeuronList_.at(i) < numNeurons);
      starterMap_[endogenouslyActiveNeuronList_.at(i)] = true;
   }
}
