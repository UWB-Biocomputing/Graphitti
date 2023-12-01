/**
 *  @file All911Edges.cpp
 *
 *  @ingroup Simulator/Edges/NG911
 *
 *  @brief Specialization of the AllEdges class for the NG911 network
 */

#include "All911Edges.h"

All911Edges::All911Edges(int numVertices, int maxEdges)
{
}

void All911Edges::setupEdges()
{
   // Setup the variables in the Super Class
   AllEdges::setupEdges();

   // Setup the variables in the sub Class
   BGSIZE maxTotalEdges = maxEdgesPerVertex_ * countVertices_;

   if (maxTotalEdges > 0) {
      isAvailable_ = make_unique<bool[]>(maxTotalEdges);
      fill_n(isAvailable_.get(), maxTotalEdges, true);

      isRedial_ = make_unique<bool[]>(maxTotalEdges);
      fill_n(isRedial_.get(), maxTotalEdges, false);

      call_.resize(maxTotalEdges);
   }
}

void All911Edges::createEdge(BGSIZE iEdg, int srcVertex, int destVertex, BGFLOAT deltaT,
                             edgeType type)
{
   inUse_[iEdg] = 1;   // True : 1 , False : 0
   destVertexIndex_[iEdg] = destVertex;
   sourceVertexIndex_[iEdg] = srcVertex;
   W_[iEdg] = 10;   // Figure this out
   this->type_[iEdg] = type;
}

#if !defined(USE_GPU)

///  Advance all the edges in the simulation.
void All911Edges::advanceEdges(AllVertices &vertices, EdgeIndexMap &edgeIndexMap)
{
   Simulator &simulator = Simulator::getInstance();
   All911Vertices &all911Vertices = dynamic_cast<All911Vertices &>(vertices);

   for (int vertex = 0; vertex < simulator.getTotalVertices(); ++vertex) {
      int start = edgeIndexMap.incomingEdgeBegin_[vertex];
      int count = edgeIndexMap.incomingEdgeCount_[vertex];

      if (simulator.getModel().getLayout().vertexTypeMap_[vertex] == CALR) {
         continue;   // TODO911: Caller Regions will have different behaviour
      }

      // Loop over all the edges and pull the data in
      for (int eIdxMap = start; eIdxMap < start + count; ++eIdxMap) {
         int edgeIdx = edgeIndexMap.incomingEdgeIndexMap_[eIdxMap];

         if (!inUse_[edgeIdx]) {
            continue;
         }   // Edge isn't in use
         if (isAvailable_[edgeIdx]) {
            continue;
         }   // Edge doesn't have a call

         int dst = destVertexIndex_[edgeIdx];
         // The destination vertex should be the one pulling the information
         assert(dst == vertex);

         CircularBuffer<Call> &dstQueue = all911Vertices.getQueue(dst);
         if (dstQueue.size() == (dstQueue.capacity() - all911Vertices.busyServers(dst))) {
            // Call is dropped because there is no space in the waiting queue
            if (!isRedial_[edgeIdx]) {
               // Only count the dropped call if it's not a redial
               all911Vertices.droppedCalls(dst)++;
               // Record that we received a call
               all911Vertices.receivedCalls(dst)++;
               LOG4CPLUS_DEBUG(edgeLogger_,
                               "Call dropped: " << all911Vertices.droppedCalls(dst) << ", time: "
                                                << call_[edgeIdx].time << ", vertex: " << dst
                                                << ", queue size: " << dstQueue.size());
            }
         } else {
            // Transfer call to destination
            dstQueue.put(call_[edgeIdx]);
            // Record that we received a call
            all911Vertices.receivedCalls(dst)++;
            isAvailable_[edgeIdx] = true;
            isRedial_[edgeIdx] = false;
         }
      }
   }
}

///  Advance one specific edge.
///
///  @param  iEdg      Index of the edge to connect to.
///  @param  vertices   The vertex list to search from.
void All911Edges::advance911Edge(BGSIZE iEdg, All911Vertices &vertices)
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