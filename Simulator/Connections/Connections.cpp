/**
 * @file Connections.cpp
 * 
 * @ingroup Simulator/Connections
 * 
 * @brief Methods for creating and updating connections
 * 
 * Below all of the resources for the various
 * connections are instantiated and initialized.
 * All of the allocation for memory is done in the
 * constructor’s parameters and not in the body of
 * the function. Once all memory has been allocated
 * the constructor fills in known information
 * into “radii” and “rates”.
 *  
 */

#include "Connections.h"
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
   edgeLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("edge"));
}

Connections::~Connections() {
}

shared_ptr<AllEdges> Connections::getEdges() const {
   return edges_;
}

shared_ptr<EdgeIndexMap> Connections::getEdgeIndexMap() const {
   return synapseIndexMap_;
}

void Connections::createEdgeIndexMap() {
   Simulator& simulator = Simulator::getInstance();
   int vertexCount = simulator.getTotalVertices();
   int maxEdges = vertexCount * edges_->maxEdgesPerVertex_;

   if (synapseIndexMap_ == nullptr) {
      synapseIndexMap_ = shared_ptr<EdgeIndexMap>(new EdgeIndexMap(vertexCount, maxEdges));
   }

   fill_n(synapseIndexMap_->incomingEdgeBegin_, vertexCount, 0);
   fill_n(synapseIndexMap_->incomingEdgeCount_, vertexCount, 0);
   fill_n(synapseIndexMap_->incomingEdgeIndexMap_, maxEdges, 0);
   fill_n(synapseIndexMap_->outgoingEdgeBegin_, vertexCount, 0);
   fill_n(synapseIndexMap_->outgoingEdgeCount_, vertexCount, 0);
   fill_n(synapseIndexMap_->outgoingEdgeIndexMap_, maxEdges, 0);

   edges_->createEdgeIndexMap(synapseIndexMap_);
}

///  Update the connections status in every epoch.
///
///  @param  vertices  The vertex list to search from.
///  @param  layout   Layout information of the neural network.
///  @return true if successful, false otherwise.
bool Connections::updateConnections(AllVertices &vertices, Layout *layout) {
   return false;
}

#if defined(USE_GPU)
void Connections::updateSynapsesWeights(const int numVertices, AllVertices &vertices, AllEdges &synapses, AllSpikingNeuronsDeviceProperties* allVerticesDevice, AllSpikingSynapsesDeviceProperties* allEdgesDevice, Layout *layout)
{
}
#else

///  Update the weight of the Synapses in the simulation.
///  Note: Platform Dependent.
///
///  @param  numVertices  Number of vertices to update.
///  @param  vertices     The vertex list to search from.
///  @param  synapses    The Synapse list to search from.
void Connections::updateSynapsesWeights(const int numVertices, AllVertices &vertices, AllEdges &synapses, Layout *layout) {
}

#endif // !USE_GPU

///  Creates synapses from synapse weights saved in the serialization file.
///
///  @param  numVertices  Number of vertices to update.
///  @param  layout      Layout information of the neural network.
///  @param  ivertices    The vertex list to search from.
///  @param  isynapses   The Synapse list to search from.
void Connections::createSynapsesFromWeights(const int numVertices, Layout *layout, AllVertices &vertices,
                                            AllEdges &synapses) {
   // for each neuron
   for (int i = 0; i < numVertices; i++) {
      // for each synapse in the vertex
      for (BGSIZE synapseIndex = 0;
           synapseIndex < Simulator::getInstance().getMaxEdgesPerVertex(); synapseIndex++) {
         BGSIZE iEdg = Simulator::getInstance().getMaxEdgesPerVertex() * i + synapseIndex;
         // if the synapse weight is not zero (which means there is a connection), create the synapse
         if (synapses.W_[iEdg] != 0.0) {
            BGFLOAT theW = synapses.W_[iEdg];
            BGFLOAT *sumPoint = &(vertices.summationMap_[i]);
            int srcVertex = synapses.sourceVertexIndex_[iEdg];
            int destVertex = synapses.destVertexIndex_[iEdg];
            edgeType type = layout->edgType(srcVertex, destVertex);
            synapses.edgeCounts_[i]++;
            synapses.createEdge(iEdg, srcVertex, destVertex, sumPoint, Simulator::getInstance().getDeltaT(),
                                   type);
            synapses.W_[iEdg] = theW;
         }
      }
   }
}



