/**
 * @file ConnStatic.cpp
 *
 * @ingroup Simulator/Connections
 * 
 * @brief The model of the small world network
 */


#include "ConnStatic.h"
#include "AllEdges.h"
#include "AllNeuroEdges.h"
#include "AllVertices.h"
#include "OperationManager.h"
#include "ParameterManager.h"
#include "ParseParamError.h"
#include "XmlRecorder.h"

#ifdef USE_HDF5
   #include "Hdf5Recorder.h"
#endif

#include <algorithm>

ConnStatic::ConnStatic()
{
   threshConnsRadius_ = 0;
   connsPerVertex_ = 0;
   rewiringProbability_ = 0;
   radiiSize_ = 0;
}

void ConnStatic::setup() {
   int added = 0;
   LOG4CPLUS_INFO(fileLogger_, "Initializing connections");

   // Obtain the Layout, which holds the vertices, from the Model
   Layout &layout = Simulator::getInstance().getModel().getLayout();
   AllVertices &vertices = layout.getVertices();

   Simulator &simulator = Simulator::getInstance();
   int numVertices = simulator.getTotalVertices();

   // Create a container for distance and destination vertex info for each vertex
   vector<DistDestVertex> distDestVertices[numVertices];

   // Max possible edges in the graph
   BGSIZE maxTotalEdges = simulator.getMaxEdgesPerVertex() * simulator.getTotalVertices();
   
   // Resize arrays to store edge data for the current epoch
   WCurrentEpoch_.resize(maxTotalEdges);
   sourceVertexIndexCurrentEpoch_.resize(maxTotalEdges);
   destVertexIndexCurrentEpoch_.resize(maxTotalEdges);

   // Reference to the neuro edge data structure
   AllNeuroEdges &neuroEdges = dynamic_cast<AllNeuroEdges &>(*edges_);

   // Store the number of vertices
   radiiSize_ = numVertices;

   // Iterate over all vertices and their connections
   for (int srcVertex = 0; srcVertex < numVertices; srcVertex++) {
       distDestVertices[srcVertex].clear();

       // Iterate over all other vertices to check if the distance is within threshold
       for (int destVertex = 0; destVertex < numVertices; destVertex++) {
           if (srcVertex != destVertex) {
               BGFLOAT dist = layout.dist_(srcVertex, destVertex);
               if (dist <= threshConnsRadius_) {
                   DistDestVertex distDestVertex {dist, destVertex};
                   distDestVertices[srcVertex].push_back(distDestVertex);
               }
           }
       }

       // Sort connections by distance (ascending order)
       sort(distDestVertices[srcVertex].begin(), distDestVertices[srcVertex].end());

       // Pick the shortest `connsPerVertex_` connections
       for (BGSIZE i = 0; i < distDestVertices[srcVertex].size() && (int)i < connsPerVertex_; i++) {
           int destVertex = distDestVertices[srcVertex][i].destVertex;
           edgeType type = layout.edgType(srcVertex, destVertex);

           // Log connection details
           LOG4CPLUS_DEBUG(fileLogger_, "Source: " << srcVertex << " Dest: " << destVertex 
                                               << " Dist: " << distDestVertices[srcVertex][i].dist);

           // Add edge to the graph
           BGSIZE iEdg = edges_->addEdge(type, srcVertex, destVertex, simulator.getDeltaT());
           added++;

           // Create NeuralEdgeProperties for this edge
           NeuralEdgeProperties edgeProps;
           edgeProps.source = std::to_string(srcVertex);
           edgeProps.target = std::to_string(destVertex);

           // Set edge weight based on connection type (excitatory or inhibitory)
           if (neuroEdges.edgSign(type) > 0) {
               edgeProps.weight = initRNG.inRange(excWeight_[0], excWeight_[1]);
           } else {
               edgeProps.weight = initRNG.inRange(inhWeight_[0], inhWeight_[1]);
           }

           // Assign the weight to the edge
           neuroEdges.W_[iEdg] = edgeProps.weight;
       }
   }

   // Log the total number of added edges
   LOG4CPLUS_DEBUG(fileLogger_, "Added connections: " << added);

   // Rewiring connections based on rewiring probability
   int nRewiring = added * rewiringProbability_;
   LOG4CPLUS_DEBUG(fileLogger_, "Rewiring connections: " << nRewiring);
}







/// Load member variables from configuration file.
/// Registered to OperationManager as Operations::op::loadParameters
void ConnStatic::loadParameters()
{
   ParameterManager::getInstance().getBGFloatByXpath("//threshConnsRadius/text()",
                                                     threshConnsRadius_);
   ParameterManager::getInstance().getIntByXpath("//connsPerNeuron/text()", connsPerVertex_);
   ParameterManager::getInstance().getBGFloatByXpath("//rewiringProbability/text()",
                                                     rewiringProbability_);
   //ParameterManager::getInstance().getBGFloatByXpath("//excWeight/min/text()", excWeight_[0]);
   //ParameterManager::getInstance().getBGFloatByXpath("//excWeight/max/text()", excWeight_[1]);
   //ParameterManager::getInstance().getBGFloatByXpath("//inhWeight/min/text()", inhWeight_[0]);
   //ParameterManager::getInstance().getBGFloatByXpath("//inhWeight/max/text()", inhWeight_[1]);
}

///  Prints out all parameters to logging file.
///  Registered to OperationManager as Operation::printParameters
void ConnStatic::printParameters() const
{
   LOG4CPLUS_DEBUG(fileLogger_, "CONNECTIONS PARAMETERS"
                                   << endl
                                   << "\tConnections Type: ConnStatic" << endl
                                   << "\tConnection radius threshold: " << threshConnsRadius_
                                   << endl
                                   << "\tConnections per neuron: " << connsPerVertex_ << endl
                                   << "\tRewiring probability: " << rewiringProbability_ << endl
                                   << "\tExhitatory min weight: " << excWeight_[0] << endl
                                   << "\tExhitatory max weight: " << excWeight_[1] << endl
                                   << "\tInhibitory min weight: " << inhWeight_[0] << endl
                                   << "\tInhibitory max weight: " << inhWeight_[1] << endl
                                   << endl);
}
