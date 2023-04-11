/**
 *  @file All911Edges.cpp
 *
 *  @ingroup Simulator/Edges/NG911
 *
 *  @brief A container of all 911 edge data
 */

#include "All911Edges.h"

All911Edges::All911Edges(const int numVertices, const int maxEdges)
{
}

void All911Edges::setupEdges()
{
   int numVertices = Simulator::getInstance().getTotalVertices();
   int maxEdges = Simulator::getInstance().getMaxEdgesPerVertex();

   BGSIZE maxTotalEdges = maxEdges * numVertices;

   maxEdgesPerVertex_ = maxEdges;
   totalEdgeCount_ = 0;
   countVertices_ = numVertices;

   // To do: Figure out whether we need all of these
   if (maxTotalEdges != 0) {
      // psr_.resize(maxTotalEdges);
      // psr_.assign(maxTotalEdges, 0.0);
      sourceVertexIndex_.resize(maxTotalEdges);
      sourceVertexIndex_.assign(maxTotalEdges, 0);

      destVertexIndex_.resize(maxTotalEdges, 0);
      destVertexIndex_.assign(maxTotalEdges, 0);

      summationPoint_.resize(maxTotalEdges, nullptr);
      summationPoint_.assign(maxTotalEdges, nullptr);

      W_.resize(maxTotalEdges, 0);
      W_.assign(maxTotalEdges, 0);

      type_.resize(maxTotalEdges);
      type_.assign(maxTotalEdges, ETYPE_UNDEF);

      edgeCounts_.resize(numVertices, 0);
      edgeCounts_.assign(numVertices, 0);

      inUse_ = make_unique<bool[]>(maxTotalEdges);
      fill_n(inUse_.get(), maxTotalEdges, false);
   }
}

void All911Edges::createEdge(const BGSIZE iEdg, int srcVertex, int destVertex, BGFLOAT *sumPoint,
                             const BGFLOAT deltaT, edgeType type)
{
   inUse_[iEdg] = true;
   summationPoint_[iEdg] = sumPoint;
   destVertexIndex_[iEdg] = destVertex;
   sourceVertexIndex_[iEdg] = srcVertex;
   W_[iEdg] = 10;   // Figure this out
   this->type_[iEdg] = type;
}

#if !defined(USE_GPU)

///  Advance all the edges in the simulation.
///
///  @param  vertices           The vertex list to search from.
///  @param  edgeIndexMap   Pointer to EdgeIndexMap structure.
void All911Edges::advanceEdges(AllVertices *vertices, EdgeIndexMap *edgeIndexMap)
{
   All911Vertices *allVertices = dynamic_cast<All911Vertices *>(vertices);
   for (BGSIZE i = 0; i < totalEdgeCount_; i++) {
      if (!inUse_[i]) {
         continue;
      }
      // if the edge is in use...
      BGSIZE iEdg = edgeIndexMap->incomingEdgeIndexMap_[i];
      advance911Edge(iEdg, allVertices);
   }
}

///  Advance one specific edge.
///
///  @param  iEdg      Index of the edge to connect to.
///  @param  vertices   The vertex list to search from.
void All911Edges::advance911Edge(const BGSIZE iEdg, All911Vertices *vertices)
{
   // edge
   // source node   -->   destination node

   // // is an input in the queue?
   // bool fPre = isSpikeQueue(iEdg);
   // bool fPost = isSpikeQueuePost(iEdg);

   // if (fPre || fPost) {

   //    BGFLOAT deltaT = Simulator::getInstance().getDeltaT();

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

#endif