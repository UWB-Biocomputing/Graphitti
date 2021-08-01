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
#include "ParameterManager.h"

// TODO: Neither the constructor nor the destructor are needed here, right?
DynamicLayout::DynamicLayout() : Layout() {
}

DynamicLayout::~DynamicLayout() {
}

///  Prints out all parameters to logging file.
///  Registered to OperationManager as Operation::printParameters
void DynamicLayout::printParameters() const {
   Layout::printParameters();
   LOG4CPLUS_DEBUG(fileLogger_, "\n\tLayout type: Dynamic Layout" << endl
                                 << "\tfraction endogenously active:" << fractionEndogenouslyActive_ << endl
                                 << "\tfraction excitatory:" << fractionExcitatory_ << endl << endl);
}

///  Creates a randomly ordered distribution with the specified numbers of neuron types.
///
///  @param  numVertices number of the vertices to have in the type map.
void DynamicLayout::generateVertexTypeMap(int numVertices) {
   LOG4CPLUS_DEBUG(fileLogger_, "\nInitializing vertex type map..." << endl);

   // Populate vertexTypeMap_ with EXC
   fill_n(vertexTypeMap_, numVertices, EXC);

   // for (int i = 0; i < numVertices; i++) {
   //    vertexTypeMap_[i] = EXC;
   // }

   int numExcitatory = (int) (fractionExcitatory_ * numVertices + 0.5);
   int numInhibitory = numVertices - numExcitatory;

   LOG4CPLUS_DEBUG(fileLogger_, "\nVERTEX TYPE MAP" << endl
                                 << "\tTotal vertices: " << numVertices << endl
                                 << "\tInhibitory Neurons: " << numInhibitory << endl
                                 << "\tExcitatory Neurons: " << numExcitatory << endl);

   LOG4CPLUS_INFO(fileLogger_, "Randomly selecting inhibitory neurons...");

   int *rgInhibitoryLayout = new int[numInhibitory];

   for (int i = 0; i < numInhibitory; i++) {
      rgInhibitoryLayout[i] = i;
   }

   for (int i = numInhibitory; i < numVertices; i++) {
      int j = static_cast<int>(initRNG() * numVertices);
      if (j < numInhibitory) {
         rgInhibitoryLayout[j] = i;
      }
   }

   for (int i = 0; i < numInhibitory; i++) {
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

   numEndogenouslyActiveNeurons_ = (BGSIZE) (fractionEndogenouslyActive_ * numVertices + 0.5);
   BGSIZE startersAllocated = 0;

   LOG4CPLUS_DEBUG(fileLogger_, "\nNEURON STARTER MAP" << endl
                                 << "\tTotal Neurons: " << numVertices << endl
                                 << "\tStarter Neurons: " << numEndogenouslyActiveNeurons_
                                 << endl);

   // randomly set neurons as starters until we've created enough
   while (startersAllocated < numEndogenouslyActiveNeurons_) {
      // Get a random neuron ID
      int i = static_cast<int>(initRNG.inRange(0, numVertices));

      // If the neuron at that index is excitatory and not already in the
      // starter map, add an entry.
      if (vertexTypeMap_[i] == EXC && !starterMap_[i]) {
         starterMap_[i] = true;
         startersAllocated++;
         LOG4CPLUS_DEBUG(fileLogger_, "Allocated EA neuron at random index [" << i << "]" << endl;);
      }
   }

   LOG4CPLUS_INFO(fileLogger_, "Done randomly initializing starter map");
}

/// Load member variables from configuration file. Registered to OperationManager as Operations::op::loadParameters
void DynamicLayout::loadParameters() {
   // Get the file paths for the Neuron lists from the configuration file
   string activeNListFilePath;
   string inhibitoryNListFilePath;
   if (!ParameterManager::getInstance().getStringByXpath("//LayoutFiles/activeNListFileName/text()",
                                                         activeNListFilePath)) {
      throw runtime_error("In Layout::loadParameters() Endogenously "
                          "active neuron list file path wasn't found and will not be initialized");
   }
   if (!ParameterManager::getInstance().getStringByXpath("//LayoutFiles/inhNListFileName/text()",
                                                         inhibitoryNListFilePath)) {
      throw runtime_error("In Layout::loadParameters() "
                          "Inhibitory neuron list file path wasn't found and will not be initialized");
   }

   // Initialize Neuron Lists based on the data read from the xml files
   if (!ParameterManager::getInstance().getIntVectorByXpath(activeNListFilePath, "A", endogenouslyActiveNeuronList_)) {
      throw runtime_error("In Layout::loadParameters() "
                          "Endogenously active neuron list file wasn't loaded correctly"
                          "\n\tfile path: " + activeNListFilePath);
   }
   numEndogenouslyActiveNeurons_ = endogenouslyActiveNeuronList_.size();
   if (!ParameterManager::getInstance().getIntVectorByXpath(inhibitoryNListFilePath, "I", inhibitoryNeuronLayout_)) {
      throw runtime_error("In Layout::loadParameters() "
                          "Inhibitory neuron list file wasn't loaded correctly."
                          "\n\tfile path: " + inhibitoryNListFilePath);
   }
}

///  Returns the type of synapse at the given coordinates
///
///  @param    srcVertex  integer that points to a Neuron in the type map as a source.
///  @param    destVertex integer that points to a Neuron in the type map as a destination.
///  @return type of the synapse.
edgeType DynamicLayout::edgType(const int srcVertex, const int destVertex) {
   if (vertexTypeMap_[srcVertex] == INH && vertexTypeMap_[destVertex] == INH)
      return II;
   else if (vertexTypeMap_[srcVertex] == INH && vertexTypeMap_[destVertex] == EXC)
      return IE;
   else if (vertexTypeMap_[srcVertex] == EXC && vertexTypeMap_[destVertex] == INH)
      return EI;
   else if (vertexTypeMap_[srcVertex] == EXC && vertexTypeMap_[destVertex] == EXC)
      return EE;

   return ETYPE_UNDEF;
}
