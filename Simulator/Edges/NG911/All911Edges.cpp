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

void All911Edges::printParameters() const {

}

void All911Edges::advanceEdge(const BGSIZE iEdg, IAllVertices *vertices) {

}

///  Get the sign of the edgeType.
///
///  @param    type    edgeType 
///  @return   1 or -1, or 0 if error
int All911Edges::edgSign(const edgeType type) {
   // switch (type) {
   //    case CP:
   //    case PR:
   //       return -1;
   //    case ETYPE_UNDEF:
   //       // TODO error.
   //       return 0;
   // }
   return 0;
}
