/**
 * @file DynamicLayout.cpp
 *
 * @ingroup Simulation/Layouts
 * 
 * @brief 
 */

#include "DynamicLayout.h"
#include "ParseParamError.h"
#include "Util.h"

DynamicLayout::DynamicLayout() : Layout() {
}

DynamicLayout::~DynamicLayout() {
}

///  Prints out all parameters to logging file.
///  Registered to OperationManager as Operation::printParameters
void DynamicLayout::printParameters() const {
   Layout::printParameters();
   LOG4CPLUS_DEBUG(fileLogger_, "\n\tLayout type: Dynamic Layout" << endl
                                       << "\tfrac_EXC:" << m_frac_excitatory_neurons << endl
                                       << "\tStarter neurons:" << m_frac_starter_neurons << endl << endl);
}

///  Creates a randomly ordered distribution with the specified numbers of neuron types.
///
///  @param  numNeurons number of the neurons to have in the type map.
void DynamicLayout::generateNeuronTypeMap(int numNeurons) {
   Layout::generateNeuronTypeMap(numNeurons);

   int numExcititoryNeurons = (int) (m_frac_excitatory_neurons * numNeurons + 0.5);
   int numInhibitoryNeurons = numNeurons - numExcititoryNeurons;

   LOG4CPLUS_DEBUG(fileLogger_, "\nNEURON TYPE MAP" << endl
                                                    << "\tTotal neurons: " << numNeurons << endl
                                                    << "\tInhibitory Neurons: " << numInhibitoryNeurons << endl
                                                    << "\tExcitatory Neurons: " << numExcititoryNeurons << endl);

   LOG4CPLUS_INFO(fileLogger_, "Randomly selecting inhibitory neurons...");

   int *rgInhibitoryLayout = new int[numInhibitoryNeurons];

   for (int i = 0; i < numInhibitoryNeurons; i++) {
      rgInhibitoryLayout[i] = i;
   }

   for (int i = numInhibitoryNeurons; i < numNeurons; i++) {
      int j = static_cast<int>(rng() * numNeurons);
      if (j < numInhibitoryNeurons) {
         rgInhibitoryLayout[j] = i;
      }
   }

   for (int i = 0; i < numInhibitoryNeurons; i++) {
      neuronTypeMap_[rgInhibitoryLayout[i]] = INH;
   }
   delete[] rgInhibitoryLayout;

   LOG4CPLUS_INFO(fileLogger_, "Done initializing neuron type map");
}

///  Populates the starter map.
///  Selects numEndogenouslyActiveNeurons_ excitatory neurons
///  and converts them into starter neurons.
///
///  @param  numNeurons number of neurons to have in the map.
void DynamicLayout::initStarterMap(const int numNeurons) {
   Layout::initStarterMap(numNeurons);

   numEndogenouslyActiveNeurons_ = (BGSIZE) (m_frac_starter_neurons * numNeurons + 0.5);
   BGSIZE startersAllocated = 0;

   LOG4CPLUS_DEBUG(fileLogger_, "\nNEURON STARTER MAP" << endl
                                                       << "\tTotal Neurons: " << numNeurons << endl
                                                       << "\tStarter Neurons: " << numEndogenouslyActiveNeurons_
                                                       << endl);

   // randomly set neurons as starters until we've created enough
   while (startersAllocated < numEndogenouslyActiveNeurons_) {
      // Get a random integer
      int i = static_cast<int>(rng.inRange(0, numNeurons));

      // If the neuron at that index is excitatory and a starter map
      // entry does not already exist, add an entry.
      if (neuronTypeMap_[i] == EXC && starterMap_[i] == false) {
         starterMap_[i] = true;
         startersAllocated++;
         LOG4CPLUS_DEBUG(fileLogger_, "Allocated EA neuron at random index [" << i << "]" << endl;);
      }
   }

   LOG4CPLUS_INFO(fileLogger_, "Done randomly initializing starter map");
}
