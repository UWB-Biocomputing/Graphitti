/**
 * @file ConnStatic.cpp
 *
 * @ingroup Simulator/Connections
 * 
 * @brief The model of the small world network
 */


#include "ConnStatic.h"
#include "ParseParamError.h"
#include "IAllEdges.h"
#include "AllVertices.h"
#include "AllEdges.h"
#include "OperationManager.h"

#include "XmlRecorder.h"

#ifdef USE_HDF5
#include "Hdf5Recorder.h"
#endif

#include <algorithm>

ConnStatic::ConnStatic() {
   threshConnsRadius_ = 0;
   connsPerVertex_ = 0;
   rewiringProbability_ = 0;
}

ConnStatic::~ConnStatic() {
   
}

///  Setup the internal structure of the class (allocate memories and initialize them).
///  Initialize the small world network characterized by parameters: 
///  number of maximum connections per vertex, connection radius threshold, and
///  small-world rewiring probability.
///
///  @param  layout    Layout information of the neural network.
///  @param  vertices   The Vertex list to search from.
///  @param  edges  The Synapse list to search from.
void ConnStatic::setupConnections(Layout *layout, IAllVertices *vertices, IAllEdges *edges) {
   int numVertices = Simulator::getInstance().getTotalVertices();
   vector<DistDestVertex> distDestVertices[numVertices];

   int added = 0;

   LOG4CPLUS_INFO(fileLogger_, "Initializing connections");

   for (int srcVertex = 0; srcVertex < numVertices; srcVertex++) {
      distDestVertices[srcVertex].clear();

      // pick the connections shorter than threshConnsRadius
      for (int destVertex = 0; destVertex < numVertices; destVertex++) {
         if (srcVertex != destVertex) {
            BGFLOAT dist = (*layout->dist_)(srcVertex, destVertex);
            if (dist <= threshConnsRadius_) {
               DistDestVertex distDestVertex;
               distDestVertex.dist = dist;
               distDestVertex.destVertex = destVertex;
               distDestVertices[srcVertex].push_back(distDestVertex);
            }
         }
      }

      // sort ascendant
      sort(distDestVertices[srcVertex].begin(), distDestVertices[srcVertex].end());
      // pick the shortest m_nConnsPerNeuron connections
      for (BGSIZE i = 0; i < distDestVertices[srcVertex].size() && (int) i < connsPerVertex_; i++) {
         int destVertex = distDestVertices[srcVertex][i].destVertex;
         synapseType type = layout->synType(srcVertex, destVertex);
         BGFLOAT *sumPoint = &(dynamic_cast<AllVertices *>(vertices)->summationMap_[destVertex]);

         LOG4CPLUS_DEBUG(fileLogger_, "Source: " << srcVertex << " Dest: " << destVertex << " Dist: "
                                                 << distDestVertices[srcVertex][i].dist);

         BGSIZE iEdg;
         edges->addEdge(iEdg, type, srcVertex, destVertex, sumPoint, Simulator::getInstance().getDeltaT());
         added++;

         // set edge weight
         // TODO: we need another synaptic weight distibution mode (normal distribution)
         if (edges->edgSign(type) > 0) {
            dynamic_cast<AllEdges *>(edges)->W_[iEdg] = rng.inRange(excWeight_[0], excWeight_[1]);
         } else {
            dynamic_cast<AllEdges *>(edges)->W_[iEdg] = rng.inRange(inhWeight_[0], inhWeight_[1]);
         }
      }
   }

   int nRewiring = added * rewiringProbability_;

   LOG4CPLUS_DEBUG(fileLogger_,"Rewiring connections: " << nRewiring);

   LOG4CPLUS_DEBUG(fileLogger_,"Added connections: " << added);
}

/// Load member variables from configuration file.
/// Registered to OperationManager as Operations::op::loadParameters
void ConnStatic::loadParameters() {
   // ConnStatic doesn't have any parameters to load from the configuration file.
}

///  Prints out all parameters to logging file.
///  Registered to OperationManager as Operation::printParameters
void ConnStatic::printParameters() const {
   LOG4CPLUS_DEBUG(fileLogger_, "CONNECTIONS PARAMETERS" << endl
    << "\tConnections Type: ConnStatic" << endl
    << "\tConnection radius threshold: " << threshConnsRadius_ << endl
    << "\tConnections per vertex: " << connsPerVertex_ << endl
    << "\tRewiring probability: " << rewiringProbability_ << endl << endl);
}

