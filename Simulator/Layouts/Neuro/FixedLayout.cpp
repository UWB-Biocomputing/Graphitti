/**
 * @file FixedLayout.cpp
 *
 * @ingroup Simulator/Layouts
 * 
 * @brief The Layout class defines the layout of vertices in neural networks
 */

#include "FixedLayout.h"
#include "ParseParamError.h"
#include "Util.h"
#include "ParameterManager.h"

// TODO: I don't think that either of the constructor or destructor is needed here
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

///  Creates a randomly ordered distribution with the specified numbers of vertex types.
///
///  @param  numVertices number of the vertices to have in the type map.
void FixedLayout::generateVertexTypeMap(int numVertices) {
   LOG4CPLUS_DEBUG(fileLogger_, "\nInitializing vertex type map" << endl);

   // Populate vertexTypeMap_ with EXC
   fill_n(vertexTypeMap_, numVertices, EXC);

   // for (int i = 0; i < numVertices; i++) {
   //    vertexTypeMap_[i] = EXC;
   // }

   int numInhibitoryNeurons = inhibitoryNeuronLayout_.size();
   int numExcititoryNeurons = numVertices - numInhibitoryNeurons;

   LOG4CPLUS_DEBUG(fileLogger_, "\nVERTEX TYPE MAP" << endl
   << "\tTotal vertices: " << numVertices << endl
   << "\tInhibitory Neurons: " << numInhibitoryNeurons << endl
   << "\tExcitatory Neurons: " << numExcititoryNeurons << endl);

   for (int i = 0; i < numInhibitoryNeurons; i++) {
      assert(inhibitoryNeuronLayout_.at(i) < numVertices);
      vertexTypeMap_[inhibitoryNeuronLayout_.at(i)] = INH;
   }

   LOG4CPLUS_INFO(fileLogger_, "Finished initializing vertex type map");
}

///  Populates the starter map.
///  Selects \e numStarter excitory neurons and converts them into starter neurons.
///  @param  numVertices number of vertices to have in the map.
void FixedLayout::initStarterMap(const int numVertices) {
   Layout::initStarterMap(numVertices);

   for (BGSIZE i = 0; i < numEndogenouslyActiveNeurons_; i++) {
      assert(endogenouslyActiveNeuronList_.at(i) < numVertices);
      starterMap_[endogenouslyActiveNeuronList_.at(i)] = true;
   }
}

/// Load member variables from configuration file. Registered to OperationManager as Operations::op::loadParameters
void FixedLayout::loadParameters() {
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
edgeType FixedLayout::edgType(const int srcVertex, const int destVertex) {
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
