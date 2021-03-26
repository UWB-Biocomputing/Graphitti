/**
 * @file Connections.cpp
 * 
 * @ingroup Simulator/Connections
 * 
 * @brief Methods for creating and updating connections
 *  ------------- CONNECTIONS STRUCT ------------ 
 * Below all of the resources for the various
 * connections are instantiated and initialized.
 * All of the allocation for memory is done in the
 * constructor’s parameters and not in the body of
 * the function. Once all memory has been allocated
 * the constructor fills in known information
 * into “radii” and “rates”.
 * --------------------------------------------- 
 */

#include "Connections.h"
#include "IAllEdges.h"
#include "IAllVertices.h"
#include "AllEdges.h"
#include "AllVertices.h"
#include "OperationManager.h"
#include "ParameterManager.h"
#include "EdgesFactory.h"

Connections::Connections() {
   // Create Edges/Synapses class using type definition in configuration file
   string type;
   ParameterManager::getInstance().getStringByXpath("//EdgesParams/@class", type);
   edges_ = EdgesFactory::getInstance()->createEdges(type);

   // Register printParameters function as a printParameters operation in the OperationManager
   function<void()> printParametersFunc = bind(&Connections::printParameters, this);
   OperationManager::getInstance().registerOperation(Operations::printParameters, printParametersFunc);

   // Register loadParameters function with Operation Manager
   function<void()> function = std::bind(&Connections::loadParameters, this);
   OperationManager::getInstance().registerOperation(Operations::op::loadParameters, function);

   // Get a copy of the file logger to use log4cplus macros
   fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));
}

Connections::~Connections() {
}

shared_ptr<IAllEdges> Connections::getEdges() const {
   return edges_;
}

shared_ptr<EdgeIndexMap> Connections::getSynapseIndexMap() const {
   return synapseIndexMap_;
}

void Connections::createEdgeIndexMap() {
   synapseIndexMap_ = shared_ptr<EdgeIndexMap>(edges_->createEdgeIndexMap());
}

///  Update the connections status in every epoch.
///
///  @param  vertices  The vertex list to search from.
///  @param  layout   Layout information of the neural network.
///  @return true if successful, false otherwise.
bool Connections::updateConnections(IAllVertices &vertices, Layout *layout) {
   return false;
}

#if defined(USE_GPU)
void Connections::updateSynapsesWeights(const int numVertices, IAllVertices &vertices, IAllEdges &synapses, AllSpikingNeuronsDeviceProperties* allVerticesDevice, AllSpikingSynapsesDeviceProperties* allEdgesDevice, Layout *layout)
{
}
#else

///  Update the weight of the Synapses in the simulation.
///  Note: Platform Dependent.
///
///  @param  numVertices  Number of vertices to update.
///  @param  vertices     The vertex list to search from.
///  @param  synapses    The Synapse list to search from.
void Connections::updateSynapsesWeights(const int numVertices, IAllVertices &vertices, IAllEdges &synapses, Layout *layout) {
}

#endif // !USE_GPU

///  Creates synapses from synapse weights saved in the serialization file.
///
///  @param  numVertices  Number of vertices to update.
///  @param  layout      Layout information of the neural network.
///  @param  ivertices    The vertex list to search from.
///  @param  isynapses   The Synapse list to search from.
void Connections::createSynapsesFromWeights(const int numVertices, Layout *layout, IAllVertices &ivertices,
                                            IAllEdges &isynapses) {
   AllVertices &vertices = dynamic_cast<AllVertices &>(ivertices);
   AllEdges &synapses = dynamic_cast<AllEdges &>(isynapses);

   // for each neuron
   for (int iNeuron = 0; iNeuron < numVertices; iNeuron++) {
      // for each synapse in the vertex
      for (BGSIZE synapseIndex = 0;
           synapseIndex < Simulator::getInstance().getMaxEdgesPerVertex(); synapseIndex++) {
         BGSIZE iEdg = Simulator::getInstance().getMaxEdgesPerVertex() * iNeuron + synapseIndex;
         // if the synapse weight is not zero (which means there is a connection), create the synapse
         if (synapses.W_[iEdg] != 0.0) {
            BGFLOAT theW = synapses.W_[iEdg];
            BGFLOAT *sumPoint = &(vertices.summationMap_[iNeuron]);
            int srcVertex = synapses.sourceNeuronIndex_[iEdg];
            int destVertex = synapses.destNeuronIndex_[iEdg];
            synapseType type = layout->synType(srcVertex, destVertex);
            synapses.synapseCounts_[iNeuron]++;
            synapses.createEdge(iEdg, srcVertex, destVertex, sumPoint, Simulator::getInstance().getDeltaT(),
                                   type);
            synapses.W_[iEdg] = theW;
         }
      }
   }
}



