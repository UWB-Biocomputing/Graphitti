/**
 * @file Connections.cpp
 * 
 * @ingroup Simulator/Connections
 * 
 * @brief Methods for creating and updating connections
 * 
 * Below all of the resources for the various
 * connections are instantiated and initialized.
 * All of the allocations for memory are done in the
 * constructor’s parameters and not in the body of
 * the function. Once all memory has been allocated
 * the constructor fills in known information
 * into “radii” and “rates”.
 *  
 */

#include "Connections.h"
#include "AllEdges.h"
#include "AllVertices.h"
#include "Factory.h"
#include "OperationManager.h"
#include "ParameterManager.h"

Connections::Connections()
{
   // Create Edges/Synapses class using type definition in configuration file
   string type;
   ParameterManager::getInstance().getStringByXpath("//EdgesParams/@class", type);
   edges_ = Factory<AllEdges>::getInstance().createType(type);

   // Get pointer to operations manager Singleton
   OperationManager &opsManager = OperationManager::getInstance();

   // Register printParameters function as a printParameters operation in the OperationManager
   function<void()> printParametersFunc = bind(&Connections::printParameters, this);
   opsManager.registerOperation(Operations::printParameters, printParametersFunc);

   // Register loadParameters function with Operation Manager
   function<void()> loadParamsFunc = bind(&Connections::loadParameters, this);
   opsManager.registerOperation(Operations::op::loadParameters, loadParamsFunc);

   // Register registerGraphProperties as Operations registerGraphProperties
   function<void()> regGraphPropsFunc = bind(&Connections::registerGraphProperties, this);
   opsManager.registerOperation(Operations::registerGraphProperties, regGraphPropsFunc);

   // Get a copy of the file logger to use log4cplus macros
   fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));
   edgeLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("edge"));
}

AllEdges &Connections::getEdges() const
{
   return *edges_;
}

EdgeIndexMap &Connections::getEdgeIndexMap() const
{
   return *synapseIndexMap_;
}

void Connections::registerGraphProperties()
{
   // TODO: Here we need to register the edge properties that are common
   // to all models with the GraphManager. Perhaps none.
   // This empty Base class implementation is here because Neural model
   // doesn't currently use GraphManager.
}

void Connections::createEdgeIndexMap()
{
   Simulator &simulator = Simulator::getInstance();
   int vertexCount = simulator.getTotalVertices();
   int maxEdges = vertexCount * edges_->maxEdgesPerVertex_;

   if (synapseIndexMap_ == nullptr) {
      synapseIndexMap_ = make_unique<EdgeIndexMap>(EdgeIndexMap(vertexCount, maxEdges));
   }
   edges_->createEdgeIndexMap(*synapseIndexMap_);
}

///  Update the connections status in every epoch.
///
///  @param  vertices  The vertex list to search from.
///  @return true if successful, false otherwise.
bool Connections::updateConnections(AllVertices &vertices)
{
   return false;
}

#if defined(USE_GPU)
void Connections::updateEdgesWeights(int numVertices, AllVertices &vertices, AllEdges &synapses,
                                     AllVerticesDeviceProperties *allVerticesDevice,
                                     AllEdgesDeviceProperties *allEdgesDevice, Layout &layout)
{
}
#else

///  Update the weight of the Synapses in the simulation.
///  Note: Platform Dependent.
void Connections::updateEdgesWeights()
{
}
#endif   // !USE_GPU
