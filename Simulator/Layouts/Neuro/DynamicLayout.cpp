/**
 * @file DynamicLayout.cpp
 *
 * @ingroup Simulator/Layouts
 * 
 * @brief The DynamicLayout class defines the layout of neurons in neural networks
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
///  @param  numVertices number of the vertices to have in the type map.
void DynamicLayout::generateVertexTypeMap(int numVertices) {
   Layout::generateVertexTypeMap(numVertices);

   int numExcititoryNeurons = (int) (m_frac_excitatory_neurons * numVertices + 0.5);
   int numInhibitoryNeurons = numVertices - numExcititoryNeurons;

   LOG4CPLUS_DEBUG(fileLogger_, "\nVERTEX TYPE MAP" << endl
                                                    << "\tTotal vertices: " << numVertices << endl
                                                    << "\tInhibitory Neurons: " << numInhibitoryNeurons << endl
                                                    << "\tExcitatory Neurons: " << numExcititoryNeurons << endl);

   LOG4CPLUS_INFO(fileLogger_, "Randomly selecting inhibitory neurons...");

   int *rgInhibitoryLayout = new int[numInhibitoryNeurons];

   for (int i = 0; i < numInhibitoryNeurons; i++) {
      rgInhibitoryLayout[i] = i;
   }

   for (int i = numInhibitoryNeurons; i < numVertices; i++) {
      int j = static_cast<int>(rng() * numVertices);
      if (j < numInhibitoryNeurons) {
         rgInhibitoryLayout[j] = i;
      }
   }

   for (int i = 0; i < numInhibitoryNeurons; i++) {
      vertexTypeMap_[rgInhibitoryLayout[i]] = INH;
   }
   delete[] rgInhibitoryLayout;

   LOG4CPLUS_INFO(fileLogger_, "Done initializing vertex type map");
}

///  Populates the starter map.
///  Selects numEndogenouslyActiveNeurons_ excitatory neurons
///  and converts them into starter neurons.
///
///  @param  numVertices number of vertices to have in the map.
void DynamicLayout::initStarterMap(const int numVertices) {
   Layout::initStarterMap(numVertices);

   numEndogenouslyActiveNeurons_ = (BGSIZE) (m_frac_starter_neurons * numVertices + 0.5);
   BGSIZE startersAllocated = 0;

   LOG4CPLUS_DEBUG(fileLogger_, "\nNEURON STARTER MAP" << endl
                                                       << "\tTotal Neurons: " << numVertices << endl
                                                       << "\tStarter Neurons: " << numEndogenouslyActiveNeurons_
                                                       << endl);

   // randomly set neurons as starters until we've created enough
   while (startersAllocated < numEndogenouslyActiveNeurons_) {
      // Get a random integer
      int i = static_cast<int>(rng.inRange(0, numVertices));

      // If the neuron at that index is excitatory and a starter map
      // entry does not already exist, add an entry.
      if (vertexTypeMap_[i] == EXC && starterMap_[i] == false) {
         starterMap_[i] = true;
         startersAllocated++;
         LOG4CPLUS_DEBUG(fileLogger_, "Allocated EA neuron at random index [" << i << "]" << endl;);
      }
   }

   LOG4CPLUS_INFO(fileLogger_, "Done randomly initializing starter map");
}
