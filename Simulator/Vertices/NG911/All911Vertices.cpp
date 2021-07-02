/**
 * @file All911Vertices.cpp
 * 
 * @ingroup Simulator/Vertices/NG911
 *
 * @brief A container of all 911 vertex data
 */

#include "All911Vertices.h"
#include "All911Edges.h"
#include "ParameterManager.h"
#include "Layout911.h"
#include "Connections.h"

All911Vertices::All911Vertices() {
   callNum_ = nullptr;
   callSrc_ = nullptr;
   callTime_ = nullptr;
   count = nullptr;
}

All911Vertices::~All911Vertices() {
   if (size_ != 0) {
      for (int i = 0; i < size_; i++) {
         delete[] callSrc_[i];
         delete[] callTime_[i];

         callSrc_[i] = nullptr;
         callTime_[i] = nullptr;
      }

      delete[] callSrc_;
      delete[] callTime_;
      delete[] callNum_;
      delete[] count;
   }

   callSrc_ = nullptr;
   callNum_ = nullptr;
   callNum_ = nullptr;
   count = nullptr;
}

// Allocate memory for all class properties
void All911Vertices::setupVertices() {
   AllVertices::setupVertices();

   callNum_ = new int[size_];
   count = new int[size_];
   callSrc_ = new int*[size_];
   callTime_ = new int*[size_];

   // Populate arrays with 0
   fill_n(callNum_, size_, 0);
   fill_n(count, size_, 0);
   fill_n(callSrc_, size_, nullptr);
   fill_n(callTime_, size_, nullptr);
}

// Generate callNum_ and dispNum_ for all caller and psap nodes
void All911Vertices::createAllVertices(Layout *layout) {
   vector<int> psapList;
   vector<int> respList;
   psapList.clear();
   respList.clear();

   int callersPerZone[] = {0, 0, 0, 0};
   int respPerZone[] = {0, 0, 0, 0};
    
   Layout911 *layout911 = dynamic_cast<Layout911 *>(layout);

   for (int i = 0; i < size_; i++) {  
      // Create all callers
      if (layout->vertexTypeMap_[i] == CALR) {
         callNum_[i] = rng.inRange(callNumRange_[0], callNumRange_[1]);
         callersPerZone[layout911->zone(i)] += callNum_[i];
      }

      // Find all PSAPs
      if(layout->vertexTypeMap_[i] == PSAP) {
         psapList.push_back(i);
      }

      // Find all resps
      if(layout->vertexTypeMap_[i] == RESP) {
         respList.push_back(i);
         respPerZone[layout911->zone(i)] += 1;
      }
   }

   int resource_count = 0;

   // Create all psaps
   // Dispatchers in a psap = [callers in the zone * k] + some randomness
   for (int i = 0; i < psapList.size(); i++) {
      int psapQ = layout911->zone(i);
      int dispCount = (callersPerZone[psapQ] * dispNumScale_) + rng.inRange(-5, 5);
      if (dispCount < 1) { dispCount = 1; }
      callNum_[psapList[i]] = dispCount;

      if (dispCount > resource_count) { resource_count = dispCount; }
   }

   // Create all responders
   // Responders in a node = [callers in the zone * k]/[number of responder nodes] + some randomness
   for (int i = 0; i < respList.size(); i++) {
      int respQ = layout911->zone(respList[i]);
      int respCount = (callersPerZone[respQ] * respNumScale_)/respPerZone[respQ] + rng.inRange(-5, 5);
      if (respCount < 1) { respCount = 1; }
      callNum_[respList[i]] = respCount;

      if (respCount > resource_count) { resource_count = respCount; }
   }

   // Create and populate callSrc and callTime.
   // Done over here because we need to know how many spots to allocate

   for(int i = 0; i < size_; i++) {
      callSrc_[i] = new int[resource_count];
      callTime_[i] = new int[resource_count];

      fill_n(callSrc_[i], size_, 0);
      fill_n(callTime_[i], size_, 0);
   }
}

void All911Vertices::loadParameters() {
   ParameterManager::getInstance().getIntByXpath("//CallNum/min/text()", callNumRange_[0]);
   ParameterManager::getInstance().getIntByXpath("//CallNum/max/text()", callNumRange_[1]);
   ParameterManager::getInstance().getBGFloatByXpath("//DispNumScale/text()", dispNumScale_);
   ParameterManager::getInstance().getBGFloatByXpath("//RespNumScale/text()", respNumScale_);
}


void All911Vertices::printParameters() const {

}

string All911Vertices::toString(const int index) const {
   return nullptr; // Change this
}

#if !defined(USE_GPU)

///  Update internal state of the indexed vertex (called by every simulation step).
///  Notify outgoing edges if vertex has fired.
///
///  @param  edges         The edge list to search from.
///  @param  edgeIndexMap  Reference to the EdgeIndexMap.
void All911Vertices::advanceVertices(AllEdges &edges, const EdgeIndexMap *edgeIndexMap) {
   // casting all911Edges for this method to use & modify
   All911Edges &allEdges = dynamic_cast<All911Edges &>(edges);
   shared_ptr<Layout> layout = Simulator::getInstance().getModel()->getLayout();

   // For each vertex in the network
   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      switch(layout->vertexTypeMap_[i]) {
         case PSAP:   advancePSAP(i);
            break;
         case RESP:   advanceRESP(i, edgeIndexMap, allEdges);
            break;
         case CALR:   advanceCALR(i, edgeIndexMap, allEdges);
            break;
         default:
            break;
      }
   }
}

///  Advance a PSAP node. Controls the redirection and handling of calls
///
///  @param  index         Index of the PSAP node
///  @param  edgeIndexMap  Reference to the EdgeIndexMap.
///  @param  allEdges      Reference to an instance of All911Edges
void All911Vertices::advancePSAP(const int index) {
   // 911TODO
}

///  Advance a CALR node. Generates calls and records received responses
///
///  @param  index         Index of the CALR node
///  @param  edgeIndexMap  Reference to the EdgeIndexMap.
///  @param  allEdges      Reference to an instance of All911Edges
void All911Vertices::advanceCALR(const int index, const EdgeIndexMap *edgeIndexMap, All911Edges &allEdges) {
   BGFLOAT probability = rng();

   // Decided not to spike
   if (probability < 0.5) { return; }

   // Fire
   // (probability * 2) is the same as calling rng(). Saves an expensive call
   BGSIZE output = probability * 2 * callNum_[index];

   // No need to iterate, as each CALR is only connected to one PSAP
   BGSIZE start = edgeIndexMap->outgoingEdgeBegin_[index];
   BGSIZE iEdg = edgeIndexMap->outgoingEdgeIndexMap_[start];

   // Inform Edge of new call
   if (allEdges.available[iEdg]) {
      allEdges.callSrc_[iEdg] = index;
      allEdges.callTime_[iEdg] = Simulator::getInstance().getCurrentStep();
      allEdges.available[iEdg] = false;
   } else {
      // 911TODO
      // Record a missed call
   }

   // 911TODO
   // Record received response
   count[index] = 0;
}

///  Advance a RESP node. Receives calls and sends response back to the source
///
///  @param  index         Index of the RESP node
///  @param  edgeIndexMap  Reference to the EdgeIndexMap.
///  @param  allEdges      Reference to an instance of All911Edges
void All911Vertices::advanceRESP(const int index, const EdgeIndexMap *edgeIndexMap, All911Edges &allEdges) {
   for (int i = 0; i < count[index]; i++) {
      int dst = callSrc_[index][i];
      int iEdgeStart = dst*allEdges.maxEdgesPerVertex_;
      int iEdg = 0;

      // If edges are indexed so that [iEdg = (allEdges.maxEdgesPerVertex_*dst) + index], this would be a lot
      //    fast, but would be bad for neuro simulations.

      // Find the relevant edge
      for (int e = iEdgeStart; e < iEdgeStart + allEdges.maxEdgesPerVertex_; e++) {
         if (allEdges.sourceVertexIndex_[e] == index) {
            iEdg = e;
            break;
         }
      }

      if (!allEdges.available[iEdg]) {
         // 911TODO
         // Record dropped call
         // Highly unlikely
         continue;
      }

      // Inform outgoing edge
      allEdges.callSrc_[iEdg] = dst;
      allEdges.callTime_[iEdg] = callTime_[index][i];
      allEdges.available[iEdg] = false;
   }

   count[index] = 0;
}
#endif