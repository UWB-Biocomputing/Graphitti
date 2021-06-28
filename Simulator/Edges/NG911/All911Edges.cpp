/**
 *  @file All911Edges.cpp
 *
 *  @ingroup Simulator/Edges/NG911
 *
 *  @brief A container of all 911 edge data
 */

#include "All911Edges.h"

All911Edges::All911Edges() : AllEdges() {

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

///  Create a edge index map.
EdgeIndexMap *All911Edges::createEdgeIndexMap() {
   int vertexCount = Simulator::getInstance().getTotalVertices();
   int totalEdgeCount = 0;

   // count the total edges
   for (int i = 0; i < vertexCount; i++) {
      assert(static_cast<int>(edgeCounts_[i]) < Simulator::getInstance().getMaxEdgesPerVertex());
      totalEdgeCount += edgeCounts_[i];
   }

   DEBUG (cout << "totalEdgeCount: " << totalEdgeCount << endl;)

   if (totalEdgeCount == 0) {
      return nullptr;
   }

   // FoundEdge allows us to sort a vector of edges based on distance so that
   // shorter distances are recorded first in the EdgeIndexMap. By doing this,
   // we don't have to traverse the entire list to find the nearest vertex
   struct FoundEdge {
      BGSIZE srcVertex;
      BGSIZE dstVertex;
      BGSIZE edg_i;
      BGFLOAT dist_;
      void findDist() {
         auto layout = Simulator::getInstance().getModel()->getLayout();
         dist_ = (*layout->dist_)(srcVertex, dstVertex); };
      bool operator<(const FoundEdge &other) const { return (this->dist_ < other.dist_); };
   };

   vector<FoundEdge> outgoingEdgeMap[vertexCount];
   vector<FoundEdge> incomingEdgeMap[vertexCount];

   BGSIZE edg_i = 0;
   int curr = 0;

   EdgeIndexMap *edgeIndexMap = new EdgeIndexMap(vertexCount, totalEdgeCount);

   // Find all edges for EdgeIndexMap
   for (int dstV = 0; dstV < vertexCount; dstV++) {
      BGSIZE edge_count = 0;
      for (int j = 0; j < Simulator::getInstance().getMaxEdgesPerVertex(); j++, edg_i++) {
         if (inUse_[edg_i] == true) {
            int srcV = sourceVertexIndex_[edg_i];
            assert(destVertexIndex_[edg_i] == dstV);

            FoundEdge temp;
            temp.srcVertex = srcV;
            temp.dstVertex = dstV;
            temp.edg_i = edg_i;
            temp.findDist();

            // incomingEdgeIndexMap_ isn't populated here as that
            // does not allow us to place them sorted
            outgoingEdgeMap[srcV].push_back(temp);
            incomingEdgeMap[dstV].push_back(temp);

            curr++;
            edge_count++;
         }
      }
      assert(edge_count == this->edgeCounts_[dstV]);
   }

   assert(totalEdgeCount == curr);
   this->totalEdgeCount_ = totalEdgeCount;

   // Sort edge lists based on distances
   for(int i = 0; i < vertexCount; i++) {
      sort(incomingEdgeMap[i].begin(), incomingEdgeMap[i].end());
      sort(outgoingEdgeMap[i].begin(), outgoingEdgeMap[i].end());
   }

   // Fill outgoing edge data into edgeMap
   curr = 0;
   for (int i = 0; i < vertexCount; i++) {
      edgeIndexMap->outgoingEdgeBegin_[i] = curr;
      edgeIndexMap->outgoingEdgeCount_[i] = outgoingEdgeMap[i].size();

      for (BGSIZE j = 0; j < outgoingEdgeMap[i].size(); j++, curr++) {
         edgeIndexMap->outgoingEdgeIndexMap_[curr] = outgoingEdgeMap[i][j].edg_i;
      }
   }

   // Fill incoming edge data into edgeMap
   curr = 0;
   for (int i = 0; i < vertexCount; i++) {
      edgeIndexMap->incomingEdgeBegin_[i] = curr;
      edgeIndexMap->incomingEdgeCount_[i] = incomingEdgeMap[i].size();

      for (BGSIZE j = 0; j < incomingEdgeMap[i].size(); j++, curr++) {
         edgeIndexMap->incomingEdgeIndexMap_[curr] = incomingEdgeMap[i][j].edg_i;
      }
   }

   return edgeIndexMap;
}

#if !defined(USE_GPU)

///  Advance all the edges in the simulation.
///
///  @param  vertices           The vertex list to search from.
///  @param  edgeIndexMap   Pointer to EdgeIndexMap structure.
void All911Edges::advanceEdges(IAllVertices *vertices, EdgeIndexMap *edgeIndexMap) {
   All911Vertices *allVertices = dynamic_cast<All911Vertices *>(vertices);
   for (BGSIZE i = 0; i < totalEdgeCount_; i++) {
      if(!inUse_[i]) { continue; }
      // if the edge is in use...
      BGSIZE iEdg = edgeIndexMap->incomingEdgeIndexMap_[i];
      advance911Edge(iEdg, allVertices);

      // layout->edgType(srcVertex, destVertex);


      // switch(layout->vertexTypeMap_[i]) {
      // case PSAP:   advancePSAP(i);
      //    break;
      // case RESP:   advanceRESP(i, layout);
      //    break;
      // case CALR:   advanceCALR(i, edgeIndexMap, allEdges);
      //    break;
      // default:
      //    break;
      // }

   }
}

///  Advance one specific edge.
///
///  @param  iEdg      Index of the edge to connect to.
///  @param  vertices   The vertex list to search from.
void All911Edges::advance911Edge(const BGSIZE iEdg, All911Vertices *vertices) { }

#endif