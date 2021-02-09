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
#include "IAllSynapses.h"
#include "IAllVertices.h"
#include "AllSynapses.h"
#include "AllVertices.h"
#include "OperationManager.h"
#include "ParameterManager.h"
#include "EdgesFactory.h"

Connections::Connections() {
   // Create Edges/Synapses class using type definition in configuration file
   string type;
   ParameterManager::getInstance().getStringByXpath("//SynapsesParams/@class", type);
   synapses_ = EdgesFactory::getInstance()->createEdges(type);

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

shared_ptr<IAllSynapses> Connections::getSynapses() const {
   return synapses_;
}

shared_ptr<SynapseIndexMap> Connections::getSynapseIndexMap() const {
   return synapseIndexMap_;
}

void Connections::createSynapseIndexMap() {
   synapseIndexMap_ = shared_ptr<SynapseIndexMap>(synapses_->createSynapseIndexMap());
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
void Connections::updateSynapsesWeights(const int numNeurons, IAllVertices &vertices, IAllSynapses &synapses, AllSpikingNeuronsDeviceProperties* allNeuronsDevice, AllSpikingSynapsesDeviceProperties* allSynapsesDevice, Layout *layout)
{
}
#else

///  Update the weight of the Synapses in the simulation.
///  Note: Platform Dependent.
///
///  @param  numNeurons  Number of neurons to update.
///  @param  vertices     The vertex list to search from.
///  @param  synapses    The Synapse list to search from.
void Connections::updateSynapsesWeights(const int numNeurons, IAllVertices &vertices, IAllSynapses &synapses, Layout *layout) {
}

#endif // !USE_GPU

///  Creates synapses from synapse weights saved in the serialization file.
///
///  @param  numNeurons  Number of neurons to update.
///  @param  layout      Layout information of the neural network.
///  @param  ivertices    The vertex list to search from.
///  @param  isynapses   The Synapse list to search from.
void Connections::createSynapsesFromWeights(const int numNeurons, Layout *layout, IAllVertices &ivertices,
                                            IAllSynapses &isynapses) {
   AllVertices &vertices = dynamic_cast<AllVertices &>(ivertices);
   AllSynapses &synapses = dynamic_cast<AllSynapses &>(isynapses);

   // for each neuron
   for (int iNeuron = 0; iNeuron < numNeurons; iNeuron++) {
      // for each synapse in the vertex
      for (BGSIZE synapseIndex = 0;
           synapseIndex < Simulator::getInstance().getMaxSynapsesPerNeuron(); synapseIndex++) {
         BGSIZE iSyn = Simulator::getInstance().getMaxSynapsesPerNeuron() * iNeuron + synapseIndex;
         // if the synapse weight is not zero (which means there is a connection), create the synapse
         if (synapses.W_[iSyn] != 0.0) {
            BGFLOAT theW = synapses.W_[iSyn];
            BGFLOAT *sumPoint = &(vertices.summationMap_[iNeuron]);
            int srcNeuron = synapses.sourceNeuronIndex_[iSyn];
            int destNeuron = synapses.destNeuronIndex_[iSyn];
            synapseType type = layout->synType(srcNeuron, destNeuron);
            synapses.synapseCounts_[iNeuron]++;
            synapses.createSynapse(iSyn, srcNeuron, destNeuron, sumPoint, Simulator::getInstance().getDeltaT(),
                                   type);
            synapses.W_[iSyn] = theW;
         }
      }
   }
}



