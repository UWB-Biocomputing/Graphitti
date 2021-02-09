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
///  @param  synapses  The Synapse list to search from.
void ConnStatic::setupConnections(Layout *layout, IAllVertices *vertices, IAllEdges *synapses) {
   int numNeurons = Simulator::getInstance().getTotalVertices();
   vector<DistDestVertex> distDestNeurons[numNeurons];

   int added = 0;

   LOG4CPLUS_INFO(fileLogger_, "Initializing connections");

   for (int srcNeuron = 0; srcNeuron < numNeurons; srcNeuron++) {
      distDestNeurons[srcNeuron].clear();

      // pick the connections shorter than threshConnsRadius
      for (int destNeuron = 0; destNeuron < numNeurons; destNeuron++) {
         if (srcNeuron != destNeuron) {
            BGFLOAT dist = (*layout->dist_)(srcNeuron, destNeuron);
            if (dist <= threshConnsRadius_) {
               DistDestVertex distDestVertex;
               distDestVertex.dist = dist;
               distDestVertex.destNeuron = destNeuron;
               distDestNeurons[srcNeuron].push_back(distDestVertex);
            }
         }
      }

      // sort ascendant
      sort(distDestNeurons[srcNeuron].begin(), distDestNeurons[srcNeuron].end());
      // pick the shortest m_nConnsPerNeuron connections
      for (BGSIZE i = 0; i < distDestNeurons[srcNeuron].size() && (int) i < connsPerVertex_; i++) {
         int destNeuron = distDestNeurons[srcNeuron][i].destNeuron;
         synapseType type = layout->synType(srcNeuron, destNeuron);
         BGFLOAT *sumPoint = &(dynamic_cast<AllVertices *>(vertices)->summationMap_[destNeuron]);

         LOG4CPLUS_DEBUG(fileLogger_, "Source: " << srcNeuron << " Dest: " << destNeuron << " Dist: "
                                                 << distDestNeurons[srcNeuron][i].dist);

         BGSIZE iEdg;
         synapses->addEdge(iEdg, type, srcNeuron, destNeuron, sumPoint, Simulator::getInstance().getDeltaT());
         added++;

         // set synapse weight
         // TODO: we need another synaptic weight distibution mode (normal distribution)
         if (synapses->synSign(type) > 0) {
            dynamic_cast<AllEdges *>(synapses)->W_[iEdg] = rng.inRange(excWeight_[0], excWeight_[1]);
         } else {
            dynamic_cast<AllEdges *>(synapses)->W_[iEdg] = rng.inRange(inhWeight_[0], inhWeight_[1]);
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

