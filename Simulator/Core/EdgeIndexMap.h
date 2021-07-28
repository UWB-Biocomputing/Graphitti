/**
 *  @file EdgeIndexMap.h
 *
 *  @brief A structure maintains outgoing and edges list (forward map).
 *
 *  @ingroup Simulator/Core
 *
 *  The structure maintains a list of outgoing edges (forward map) and edges list.
 *
 *  The outgoing edges list stores all outgoing edge indices relevant to a vertex.
 *  Edge indices are stored in the edge forward map (forwardIndex), and
 *  the pointer and length of the vertex i's outgoing edge indices are specified
 *  by outgoingSynapse_begin[i] and edgeCount[i] respectively.
 *  The incoming edges list is used in calcSummationMapDevice() device function to
 *  calculate sum of PSRs for each vertex simultaneously.
 *  The list also used in AllSpikingNeurons::advanceVertices() function to allow back propagation.
 *
 *  The edges list stores all edge indices.
 *  The list is referred in advanceEdgesDevice() device function.
 */

#pragma once

#include "BGTypes.h"
#include <valarray>

using namespace std;

struct EdgeIndexMap {
   /// Pointer to the outgoing edge index map.
   valarray<BGSIZE> outgoingEdgeIndexMap_;

   /// The beginning index of the outgoing edge for each vertex.
   /// Indexed by a source vertex index.
   valarray<BGSIZE> outgoingEdgeBegin_;

   /// The number of outgoing edges of each vertex.
   /// Indexed by a source vertex index.
   valarray<BGSIZE> outgoingEdgeCount_;

   /// Pointer to the incoming edge index map.
   valarray<BGSIZE> incomingEdgeIndexMap_;

   /// The beginning index of the incoming edge for each vertex.
   /// Indexed by a destination vertex index.
   valarray<BGSIZE> incomingEdgeBegin_;

   /// The number of incoming edges for each vertex.
   /// Indexed by a destination vertex index.
   valarray<BGSIZE> incomingEdgeCount_;

   EdgeIndexMap() : numOfVertices_(0), numOfEdges_(0) {
   };

   EdgeIndexMap(int vertexCount, int edgeCount) : numOfVertices_(vertexCount), numOfEdges_(edgeCount) {
      if (numOfVertices_ > 0) {
         outgoingEdgeBegin_ = valarray<BGSIZE>(numOfVertices_);
         outgoingEdgeCount_ = valarray<BGSIZE>(numOfVertices_);
         incomingEdgeBegin_ = valarray<BGSIZE>(numOfVertices_);
         incomingEdgeCount_ = valarray<BGSIZE>(numOfVertices_);
      }

      if (numOfEdges_ > 0) {
         outgoingEdgeIndexMap_ = valarray<BGSIZE>(numOfEdges_);
         incomingEdgeIndexMap_ = valarray<BGSIZE>(numOfEdges_);
      }

      outgoingEdgeBegin_ = 0;
      outgoingEdgeCount_ = 0;
      outgoingEdgeIndexMap_ = 0;
      incomingEdgeBegin_ = 0;
      incomingEdgeCount_ = 0;
      incomingEdgeIndexMap_ = 0;
   };

   ~EdgeIndexMap() {
   }

private:
    /// Number of total vertices.
    BGSIZE numOfVertices_;

    /// Number of total edges.
    BGSIZE numOfEdges_;
};

