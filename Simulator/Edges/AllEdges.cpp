/**
 * @file AllEdges.cpp
 * 
 * @ingroup Simulator/Edges
 *
 * @brief A container of all edge data
 */

#include "AllEdges.h"
#include "AllVertices.h"
#include "OperationManager.h"

AllEdges::AllEdges() :
      totalEdgeCount_(0),
      maxEdgesPerVertex_(0),
      countVertices_(0) {
   destVertexIndex_ = nullptr;
   W_ = nullptr;
   summationPoint_ = nullptr;
   sourceVertexIndex_ = nullptr;
   type_ = nullptr;
   inUse_ = nullptr;
   edgeCounts_ = nullptr;

   // Register loadParameters function as a loadParameters operation in the
   // OperationManager. This will register the appropriate overridden method
   // for the actual (sub)class of the object being created.
   function<void()> loadParametersFunc = std::bind(&AllEdges::loadParameters, this);
   OperationManager::getInstance().registerOperation(Operations::op::loadParameters, loadParametersFunc);

   // Register printParameters function as a printParameters operation in the
   // OperationManager. This will register the appropriate overridden method
   // for the actual (sub)class of the object being created.
   function<void()> printParametersFunc = bind(&AllEdges::printParameters, this);
   OperationManager::getInstance().registerOperation(Operations::printParameters, printParametersFunc);

   fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));
   edgeLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("edge"));
}

AllEdges::AllEdges(const int numVertices, const int maxEdges) {
   setupEdges(numVertices, maxEdges);
}

AllEdges::~AllEdges() {
   BGSIZE maxTotalEdges = maxEdgesPerVertex_ * countVertices_;

   if (maxTotalEdges != 0) {
      delete[] destVertexIndex_;
      delete[] W_;
      delete[] summationPoint_;
      delete[] sourceVertexIndex_;
      delete[] type_;
      delete[] inUse_;
      delete[] edgeCounts_;
   }

   destVertexIndex_ = nullptr;
   W_ = nullptr;
   summationPoint_ = nullptr;
   sourceVertexIndex_ = nullptr;
   type_ = nullptr;
   inUse_ = nullptr;
   edgeCounts_ = nullptr;

   countVertices_ = 0;
   maxEdgesPerVertex_ = 0;
}

/// Load member variables from configuration file.
/// Registered to OperationManager as Operation::op::loadParameters
void AllEdges::loadParameters() {
   // Nothing to load from configuration file besides SynapseType in the current implementation.
}

///  Prints out all parameters to logging file.
///  Registered to OperationManager as Operation::printParameters
void AllEdges::printParameters() const {
   LOG4CPLUS_DEBUG(fileLogger_, "\nEDGES PARAMETERS" << endl
    << "\t---AllEdges Parameters---" << endl
    << "\tTotal edge counts: " << totalEdgeCount_ << endl
    << "\tMax edges per vertex: " << maxEdgesPerVertex_ << endl
    << "\tVertex count: " << countVertices_ << endl << endl);
}

///  Setup the internal structure of the class (allocate memories and initialize them).
void AllEdges::setupEdges() {
   setupEdges(Simulator::getInstance().getTotalVertices(), Simulator::getInstance().getMaxEdgesPerVertex());
}

///  Setup the internal structure of the class (allocate memories and initialize them).
///
///  @param  numVertices   Total number of vertices in the network.
///  @param  maxEdges  Maximum number of edges per vertex.
void AllEdges::setupEdges(const int numVertices, const int maxEdges) {
   BGSIZE maxTotalEdges = maxEdges * numVertices;

   maxEdgesPerVertex_ = maxEdges;
   totalEdgeCount_ = 0;
   countVertices_ = numVertices;

   if (maxTotalEdges != 0) {
      destVertexIndex_ = new int[maxTotalEdges];
      W_ = new BGFLOAT[maxTotalEdges];
      summationPoint_ = new BGFLOAT *[maxTotalEdges];
      sourceVertexIndex_ = new int[maxTotalEdges];
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

///  Sets the data for Synapse to input's data.
///
///  @param  input  istream to read from.
///  @param  iEdg   Index of the edge to set.
void AllEdges::readEdge(istream &input, const BGSIZE iEdg) {
   int synapse_type(0);

   // input.ignore() so input skips over end-of-line characters.
   input >> sourceVertexIndex_[iEdg];
   input.ignore();
   input >> destVertexIndex_[iEdg];
   input.ignore();
   input >> W_[iEdg];
   input.ignore();
   input >> synapse_type;
   input.ignore();
   input >> inUse_[iEdg];
   input.ignore();

   type_[iEdg] = edgeOrdinalToType(synapse_type);
}

///  Write the edge data to the stream.
///
///  @param  output  stream to print out to.
///  @param  iEdg    Index of the edge to print out.
void AllEdges::writeEdge(ostream &output, const BGSIZE iEdg) const {
   output << sourceVertexIndex_[iEdg] << ends;
   output << destVertexIndex_[iEdg] << ends;
   output << W_[iEdg] << ends;
   output << type_[iEdg] << ends;
   output << inUse_[iEdg] << ends;
}

///  Returns an appropriate edgeType object for the given integer.
///
///  @param  typeOrdinal    Integer that correspond with a edgeType.
///  @return the SynapseType that corresponds with the given integer.
edgeType AllEdges::edgeOrdinalToType(const int typeOrdinal) {
   switch (typeOrdinal) {
      case 0:
         return II;
      case 1:
         return IE;
      case 2:
         return EI;
      case 3:
         return EE;
      case 4:
         return CP;
      case 5:
         return PR;
      case 6:
         return RC;
      case 7:
         return PP;
      default:
         return ETYPE_UNDEF;
   }
}

///  Create a edge index map.
void AllEdges::createEdgeIndexMap(shared_ptr<EdgeIndexMap> edgeIndexMap) {
   Simulator& simulator = Simulator::getInstance();
   int vertexCount = simulator.getTotalVertices();
   int totalEdgeCount = 0;

   // count the total edges
   for (int i = 0; i < vertexCount; i++) {
      assert(static_cast<int>(edgeCounts_[i]) < simulator.getMaxEdgesPerVertex());
      totalEdgeCount += edgeCounts_[i];
   }

   LOG4CPLUS_TRACE(fileLogger_,endl<<"totalEdgeCount: in edgeIndexMap " << totalEdgeCount << endl);

   // Create vector for edge forwarding map
   vector<BGSIZE> rgEdgeEdgeIndexMap[vertexCount];

   BGSIZE edg_i = 0;
   int curr = 0;

   LOG4CPLUS_TRACE(edgeLogger_, "\nSize of edge Index Map "<< vertexCount << "," << totalEdgeCount << endl);
   
   for (int i = 0; i < vertexCount; i++) {
      BGSIZE edge_count = 0;
      edgeIndexMap->incomingEdgeBegin_[i] = curr;
      for (int j = 0; j < simulator.getMaxEdgesPerVertex(); j++, edg_i++) {
         if (inUse_[edg_i]) {
            int idx = sourceVertexIndex_[edg_i];
            rgEdgeEdgeIndexMap[idx].push_back(edg_i);

            edgeIndexMap->incomingEdgeIndexMap_[curr] = edg_i;
            curr++;
            edge_count++;
         }
      }
      
      if(edge_count != edgeCounts_[i])
      {
         LOG4CPLUS_FATAL(edgeLogger_, "\nedge_count does not match edgeCounts_" << edge_count << endl);
         throw runtime_error("createEdgeIndexMap: edge_count does not match edgeCounts_.");
      }

      edgeIndexMap->incomingEdgeCount_[i] = edge_count;
   }
   
   if(totalEdgeCount != curr)
   {
      LOG4CPLUS_FATAL(edgeLogger_,"Curr does not match the totalEdgeCount. curr are " << curr << endl);
      throw runtime_error("createEdgeIndexMap: Curr does not match the totalEdgeCount.");
   }
   totalEdgeCount_ = totalEdgeCount;
   LOG4CPLUS_DEBUG(edgeLogger_,endl<<"totalEdgeCount: " << totalEdgeCount_ << endl);
   
   edg_i = 0;
   for (int i = 0; i < vertexCount; i++) {
      edgeIndexMap->outgoingEdgeBegin_[i] = edg_i;
      edgeIndexMap->outgoingEdgeCount_[i] = rgEdgeEdgeIndexMap[i].size();

      for (BGSIZE j = 0; j < rgEdgeEdgeIndexMap[i].size(); j++, edg_i++) {
         edgeIndexMap->outgoingEdgeIndexMap_[edg_i] = rgEdgeEdgeIndexMap[i][j];
      }
   }
}


#if !defined(USE_GPU)

///  Advance all the edges in the simulation.
///
///  @param  vertices           The vertices.
///  @param  edgeIndexMap   Pointer to EdgeIndexMap structure.
void AllEdges::advanceEdges(AllVertices *vertices, EdgeIndexMap *edgeIndexMap) {
   for (BGSIZE i = 0; i < totalEdgeCount_; i++) {
      BGSIZE iEdg = edgeIndexMap->incomingEdgeIndexMap_[i];
      advanceEdge(iEdg, vertices);
   }
}

///  Remove a edge from the network.
///
///  @param  iVert    Index of a vertex to remove from.
///  @param  iEdg           Index of a edge to remove.
void AllEdges::eraseEdge(const int iVert, const BGSIZE iEdg) {
   edgeCounts_[iVert]--;
   inUse_[iEdg] = false;
   summationPoint_[iEdg] = nullptr;
   W_[iEdg] = 0;
}

#endif // !defined(USE_GPU)


///  Adds an edge to the model, connecting two Vertices.
///
///  @param  iEdg        Index of the edge to be added.
///  @param  type        The type of the edge to add.
///  @param  srcVertex  The Vertex that sends to this edge.
///  @param  destVertex The Vertex that receives from the edge.
///  @param  sumPoint   Summation point address.
///  @param  deltaT      Inner simulation step duration
void AllEdges::addEdge(BGSIZE &iEdg, edgeType type, const int srcVertex, const int destVertex, BGFLOAT *sumPoint,
                        const BGFLOAT deltaT) {
   if (edgeCounts_[destVertex] >= maxEdgesPerVertex_) {
      LOG4CPLUS_FATAL(edgeLogger_, "Vertex : " << destVertex << " ran out of space for new edges.");
      throw runtime_error("Vertex : " + destVertex + string(" ran out of space for new edges."));
   }

   // add it to the list: find first edge location for vertex destVertex
   // that isn't in use.
   for (BGSIZE i = 0; i < maxEdgesPerVertex_; i++) {
      iEdg = maxEdgesPerVertex_ * destVertex + i;
      if (!inUse_[iEdg]) {
         break;
      }
   }

   edgeCounts_[destVertex]++;

   // create an edge
   createEdge(iEdg, srcVertex, destVertex, sumPoint, deltaT, type);
}
