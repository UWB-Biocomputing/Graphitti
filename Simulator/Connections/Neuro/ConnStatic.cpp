/**
 * @file ConnStatic.cpp
 *
 * @ingroup Simulator/Connections
 * 
 * @brief The model of the small world network
 */


#include "ConnStatic.h"
#include "ParseParamError.h"
#include "AllVertices.h"
#include "AllEdges.h"
#include "AllNeuroEdges.h"
#include "OperationManager.h"
#include "ParameterManager.h"
#include "XmlRecorder.h"

#ifdef USE_HDF5
#include "Hdf5Recorder.h"
#endif

#include <algorithm>

ConnStatic::ConnStatic() {
   threshConnsRadius_ = 0;
   connsPerVertex_ = 0;
   rewiringProbability_ = 0;
   WCurrentEpoch_ = nullptr;
   destVertexIndexCurrentEpoch_ = nullptr;
   sourceVertexIndexCurrentEpoch_ = nullptr;
   radiiSize_ = 0;
}

ConnStatic::~ConnStatic() {
   if (WCurrentEpoch_ != nullptr) delete [] WCurrentEpoch_;
   WCurrentEpoch_ = nullptr;
   if (destVertexIndexCurrentEpoch_ != nullptr) delete [] destVertexIndexCurrentEpoch_;
   destVertexIndexCurrentEpoch_ = nullptr;
   if (sourceVertexIndexCurrentEpoch_ != nullptr) delete [] sourceVertexIndexCurrentEpoch_;
   sourceVertexIndexCurrentEpoch_ = nullptr;
}

///  Setup the internal structure of the class (allocate memories and initialize them).
///  Initialize the small world network characterized by parameters: 
///  number of maximum connections per vertex, connection radius threshold, and
///  small-world rewiring probability.
///
///  @param  layout    Layout information of the neural network.
///  @param  vertices   The Vertex list to search from.
///  @param  edges  The Synapse list to search from.
void ConnStatic::setupConnections(Layout *layout, AllVertices *vertices, AllEdges *edges) {
   Simulator& simulator = Simulator::getInstance();
   int numVertices = simulator.getTotalVertices();
   vector<DistDestVertex> distDestVertices[numVertices];
   BGSIZE maxTotalEdges =  simulator.getMaxEdgesPerVertex() * simulator.getTotalVertices();
   WCurrentEpoch_ = new BGFLOAT[maxTotalEdges];
   sourceVertexIndexCurrentEpoch_ = new int[maxTotalEdges];
   destVertexIndexCurrentEpoch_ = new int[maxTotalEdges];
   AllNeuroEdges *neuroEdges = dynamic_cast<AllNeuroEdges *>(edges);
   
   radiiSize_ = numVertices;
   int added = 0;

   LOG4CPLUS_INFO(fileLogger_, "Initializing connections");

   for (int srcVertex = 0; srcVertex < numVertices; srcVertex++) {
      distDestVertices[srcVertex].clear();

      // pick the connections shorter than threshConnsRadius
      for (int destVertex = 0; destVertex < numVertices; destVertex++) {
         if (srcVertex != destVertex) {
            BGFLOAT dist = (*layout->dist_)(srcVertex, destVertex);
            if (dist <= threshConnsRadius_) {
               DistDestVertex distDestVertex{dist, destVertex};
               distDestVertices[srcVertex].push_back(distDestVertex);
            }
         }
      }

      // sort ascendant
      sort(distDestVertices[srcVertex].begin(), distDestVertices[srcVertex].end());
      // pick the shortest connsPerVertex_ connections
      for (BGSIZE i = 0; i < distDestVertices[srcVertex].size() && (int) i < connsPerVertex_; i++) {
         int destVertex = distDestVertices[srcVertex][i].destVertex;
         edgeType type = layout->edgType(srcVertex, destVertex);
         BGFLOAT *sumPoint = &vertices->summationMap_[destVertex];

         LOG4CPLUS_DEBUG(fileLogger_, "Source: " << srcVertex << " Dest: " << destVertex << " Dist: "
                         << distDestVertices[srcVertex][i].dist);

         BGSIZE iEdg;
         edges->addEdge(iEdg, type, srcVertex, destVertex, sumPoint, simulator.getDeltaT());
         added++;


         // set edge weight
         // TODO: we need another synaptic weight distibution mode (normal distribution)
         if (neuroEdges->edgSign(type) > 0) {
            neuroEdges->W_[iEdg] = initRNG.inRange(excWeight_[0], excWeight_[1]);
         } else {
            neuroEdges->W_[iEdg] = initRNG.inRange(inhWeight_[0], inhWeight_[1]);
         }
      }
   }
   
   string weight_str="";
   for(int i=0; i<maxTotalEdges; i++)
   {
      WCurrentEpoch_[i] = neuroEdges->W_[i];
      sourceVertexIndexCurrentEpoch_[i] = neuroEdges->sourceVertexIndex_[i];
      destVertexIndexCurrentEpoch_[i] = neuroEdges->destVertexIndex_[i];
      
      if(WCurrentEpoch_[i]!=0) {
         // LOG4CPLUS_DEBUG(edgeLogger_,i << WCurrentEpoch_[i]);
         weight_str += to_string(WCurrentEpoch_[i]) + " ";
      }
      
   }
   // LOG4CPLUS_DEBUG(edgeLogger_, "Weights are " << weight_str);

   int nRewiring = added * rewiringProbability_;

   LOG4CPLUS_DEBUG(fileLogger_,"Rewiring connections: " << nRewiring);

   LOG4CPLUS_DEBUG(fileLogger_,"Added connections: " << added);
}

/// Load member variables from configuration file.
/// Registered to OperationManager as Operations::op::loadParameters
void ConnStatic::loadParameters() {
   ParameterManager::getInstance().getBGFloatByXpath("//threshConnsRadius/text()", threshConnsRadius_);
   ParameterManager::getInstance().getIntByXpath("//connsPerNeuron/text()", connsPerVertex_);
   ParameterManager::getInstance().getBGFloatByXpath("//rewiringProbability/text()", rewiringProbability_);
   //ParameterManager::getInstance().getBGFloatByXpath("//excWeight/min/text()", excWeight_[0]);
   //ParameterManager::getInstance().getBGFloatByXpath("//excWeight/max/text()", excWeight_[1]);
   //ParameterManager::getInstance().getBGFloatByXpath("//inhWeight/min/text()", inhWeight_[0]);
   //ParameterManager::getInstance().getBGFloatByXpath("//inhWeight/max/text()", inhWeight_[1]);
}

///  Prints out all parameters to logging file.
///  Registered to OperationManager as Operation::printParameters
void ConnStatic::printParameters() const {
   LOG4CPLUS_DEBUG(fileLogger_, "CONNECTIONS PARAMETERS" << endl
                   << "\tConnections Type: ConnStatic" << endl
                   << "\tConnection radius threshold: " << threshConnsRadius_ << endl
                   << "\tConnections per neuron: " << connsPerVertex_ << endl
                   << "\tRewiring probability: " << rewiringProbability_ << endl
                   << "\tExhitatory min weight: " << excWeight_[0] << endl
                   << "\tExhitatory max weight: " << excWeight_[1] << endl
                   << "\tInhibitory min weight: " << inhWeight_[0] << endl
                   << "\tInhibitory max weight: " << inhWeight_[1] << endl
                   << endl);
}

