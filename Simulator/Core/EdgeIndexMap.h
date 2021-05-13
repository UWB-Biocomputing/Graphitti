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

using namespace std;

struct EdgeIndexMap {
   /// Pointer to the outgoing edge index map.
   BGSIZE* outgoingEdgeIndexMap_;

   /// The beginning index of the outgoing edge for each vertex.
   /// Indexed by a source vertex index.
   BGSIZE *outgoingEdgeBegin_;

   /// The number of outgoing edges of each vertex.
   /// Indexed by a source vertex index.
   BGSIZE *outgoingEdgeCount_;

   /// Pointer to the incoming edge index map.
   BGSIZE* incomingEdgeIndexMap_;

   /// The beginning index of the incoming edge for each vertex.
   /// Indexed by a destination vertex index.
   BGSIZE *incomingEdgeBegin_;

   /// The number of incoming edges for each vertex.
   /// Indexed by a destination vertex index.
   BGSIZE *incomingEdgeCount_;

   EdgeIndexMap() : numOfVertices_(0), numOfEdges_(0) {
      outgoingEdgeBegin_ = nullptr;
      outgoingEdgeCount_ = nullptr;
      incomingEdgeBegin_ = nullptr;
      incomingEdgeCount_ = nullptr;

      outgoingEdgeIndexMap_ = nullptr;
      incomingEdgeIndexMap_ = nullptr;
   };

   EdgeIndexMap(int vertexCount, int edgeCount) : numOfVertices_(vertexCount), numOfEdges_(edgeCount) {
      if (numOfVertices_ > 0) {
         outgoingEdgeBegin_ = new BGSIZE[numOfVertices_];
         outgoingEdgeCount_ = new BGSIZE[numOfVertices_];
         incomingEdgeBegin_ = new BGSIZE[numOfVertices_];
         incomingEdgeCount_ = new BGSIZE[numOfVertices_];
      }

      if (numOfEdges_ > 0) {
         outgoingEdgeIndexMap_ = new BGSIZE[numOfEdges_];
         incomingEdgeIndexMap_ = new BGSIZE[numOfEdges_];
      }
   };

   ~EdgeIndexMap() {
      if (numOfVertices_ > 0) {
         delete[] outgoingEdgeBegin_;
         delete[] outgoingEdgeCount_;
         delete[] incomingEdgeBegin_;
         delete[] incomingEdgeCount_;
      }
      if (numOfEdges_ > 0) {
         delete[] outgoingEdgeIndexMap_;
         delete[] incomingEdgeIndexMap_;
      }
   }

private:
    /// Number of total vertices.
    BGSIZE numOfVertices_;

    /// Number of total edges.
    BGSIZE numOfEdges_;
};

