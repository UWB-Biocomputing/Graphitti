/**
* @file ConnStatic.cpp
*
* @ingroup Simulator/Connections/Neuro
* 
* @brief This class manages the Connections of the Neuro STDP network
*/

#include "ConnStatic.h"
#include "AllEdges.h"
#include "AllNeuroEdges.h"
#include "AllVertices.h"
#include "GraphManager.h"
#include "OperationManager.h"
#include "ParameterManager.h"
#include "ParseParamError.h"
#include "XmlRecorder.h"

#ifdef USE_HDF5
   #include "Hdf5Recorder.h"
#endif

#include <algorithm>

/// @brief Default constructor for ConnStatic
ConnStatic::ConnStatic()
{
}

/// @brief Set up the connections in the network
void ConnStatic::setup()
{
   int added = 0;
   LOG4CPLUS_INFO(fileLogger_, "Initializing connections");

   // Obtain Vertices and Layout from the Model
   Layout &layout = Simulator::getInstance().getModel().getLayout();
   AllVertices &vertices = layout.getVertices();
   // All Edges object for Connections
   AllNeuroEdges &neuroEdges = dynamic_cast<AllNeuroEdges &>(*edges_);


   // Iterator to traverse edges
   GraphManager<NeuralVertexProperties>::EdgeIterator ei, ei_end;
   GraphManager<NeuralVertexProperties> &gm = GraphManager<NeuralVertexProperties>::getInstance();

   // Initialize GraphManager and iterate through edges
   for (boost::tie(ei, ei_end) = gm.edges(); ei != ei_end; ++ei) {
      int srcVertex = gm.source(*ei);
      int destVertex = gm.target(*ei);
      double weight = gm.weight(*ei);
      edgeType type = layout.edgType(srcVertex, destVertex);
      BGFLOAT dist = layout.dist_(srcVertex, destVertex);

      // Debug
      // cout <<  "Source: " << srcVertex << " Dest: " << destVertex << " Dist: " << dist << " Weight: " << weight << endl;

      // Log edge data
      LOG4CPLUS_DEBUG(edgeLogger_, "Source: " << srcVertex << ", Dest: " << destVertex
                                              << ", Dist: " << dist << ", Weight: " << weight);

      // Add edge and store weight
      BGSIZE iEdg
         = edges_->addEdge(type, srcVertex, destVertex, Simulator::getInstance().getDeltaT());
      neuroEdges.W_[iEdg] = weight;
      added++;
   }

   // Log the total number of connections added
   LOG4CPLUS_DEBUG(fileLogger_, "Added connections: " << added);
}

/// @brief Register graph properties to NeuralVertexProperties
void ConnStatic::registerGraphProperties()
{
   Connections::registerGraphProperties();
   GraphManager<NeuralVertexProperties> &gm = GraphManager<NeuralVertexProperties>::getInstance();
   gm.registerProperty("source", &NeuralEdgeProperties::source);
   gm.registerProperty("target", &NeuralEdgeProperties::target);
   gm.registerProperty("weight", &NeuralEdgeProperties::weight);
}

/// @brief Loads parameters related to connections
void ConnStatic::loadParameters()
{
}

/// @brief Prints the parameters of the connection
void ConnStatic::printParameters() const
{
}

void ConnStatic::registerHistoryVariables()
{
   // Register the following variables to be recorded
   // Note: There may be potential duplicate weight, source, destination vertices
   Recorder &recorder = Simulator::getInstance().getModel().getRecorder();
   recorder.registerVariable("weight", WCurrentEpoch_, Recorder::UpdatedType::DYNAMIC);
   recorder.registerVariable("sourceVertex", sourceVertexIndexCurrentEpoch_,
                             Recorder::UpdatedType::DYNAMIC);
   recorder.registerVariable("destinationVertex", destVertexIndexCurrentEpoch_,
                             Recorder::UpdatedType::DYNAMIC);
}
