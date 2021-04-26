/**
 *  @file All911Edges.cpp
 *
 *  @ingroup Simulator/Edges/NG911
 *
 *  @brief A container of all 911 edge data
 */

#include "All911Edges.h"

All911Edges::All911Edges() {

}

All911Edges::All911Edges(const int numVertices, const int maxEdges) {

}

All911Edges::~All911Edges() {

}

void All911Edges::setupEdges() {
   int numVertices = Simulator::getInstance().getTotalVertices();
   int maxEdges = Simulator::getInstance().getMaxEdgesPerVertex();

   BGSIZE maxTotalEdges = maxEdges * numVertices;

   maxEdgesPerVertex_ = maxEdges;
   totalEdgeCount_ = 0;
   countVertices_ = numVertices;

   // To do: Figure out whether we need all of these
   if (maxTotalEdges != 0) {
      destVertexIndex_ = new int[maxTotalEdges];
      W_ = new BGFLOAT[maxTotalEdges];
      summationPoint_ = new BGFLOAT *[maxTotalEdges];
      sourceVertexIndex_ = new int[maxTotalEdges];
      // psr_ = new BGFLOAT[maxTotalEdges];
      type_ = new edgeType[maxTotalEdges];
      inUse_ = new bool[maxTotalEdges];
      edgeCounts_ = new BGSIZE[numVertices];

      for (BGSIZE i = 0; i < maxTotalEdges; i++) {
         summationPoint_[i] = nullptr;
         inUse_[i] = false;
         W_[i] = 0;
      }

      for (int i = 0; i < numVertices; i++) {
         edgeCounts_[i] = 0;
      }
   }
}

void All911Edges::createEdge(const BGSIZE iEdg, int srcVertex, int destVertex, BGFLOAT *sumPoint, const BGFLOAT deltaT,
                              edgeType type) {
   inUse_[iEdg] = true;
   summationPoint_[iEdg] = sumPoint;
   destVertexIndex_[iEdg] = destVertex;
   sourceVertexIndex_[iEdg] = srcVertex;
   W_[iEdg] = 10; // Figure this out
   this->type_[iEdg] = type;
}

///  Advance one specific edge.
///
///  @param  iEdg      Index of the edge to connect to.
///  @param  vertices   The vertex list to search from.
void All911Edges::advanceEdge(const BGSIZE iEdg, IAllVertices *vertices) {

   // // is an input in the queue?
   // bool fPre = isSpikeQueue(iEdg);
   // bool fPost = isSpikeQueuePost(iEdg);

   // if (fPre || fPost) {

   //    BGFLOAT deltaT = Simulator::getInstance().getDeltaT();
   //    All911Vertices *allVertices = dynamic_cast<All911Vertices *>(vertices);

   //    // pre and post vertices index
   //    int idxPre = sourceVertexIndex_[iEdg];
   //    int idxPost = destVertexIndex_[iEdg];
   //    uint64_t spikeHistory, spikeHistory2;
   //    BGFLOAT delta;
   //    BGFLOAT epre, epost;

   //    if (fPre) {   // preSpikeHit
   //       // spikeCount points to the next available position of spike_history,
   //       // so the getSpikeHistory w/offset = -2 will return the spike time
   //       // just one before the last spike.
   //       spikeHistory = allVertices->getSpikeHistory(idxPre, -2);
   //       if (spikeHistory != ULONG_MAX && useFroemkeDanSTDP_) {
   //          // delta will include the transmission delay
   //          delta = static_cast<BGFLOAT>(g_simulationStep - spikeHistory) * deltaT;
   //          epre = 1.0 - exp(-delta / tauspre_);
   //       } else {
   //          epre = 1.0;
   //       }

   //    }

   //    if (fPost) {   // postSpikeHit
   //       // spikeCount points to the next available position of spike_history,
   //       // so the getSpikeHistory w/offset = -2 will return the spike time
   //       // just one before the last spike.
   //       spikeHistory = allVertices->getSpikeHistory(idxPost, -2);
   //       if (spikeHistory != ULONG_MAX && useFroemkeDanSTDP_) {
   //          // delta will include the transmission delay
   //          delta = static_cast<BGFLOAT>(g_simulationStep - spikeHistory) * deltaT;
   //          epost = 1.0 - exp(-delta / tauspost_);
   //       } else {
   //          epost = 1.0;
   //       }

   //       // call the learning function stdpLearning() for each pair of
   //       // post-pre spikes
   //       int offIndex = -1;   // last spike
   //    }
   // }

}

void All911Edges::printParameters() const {

}
