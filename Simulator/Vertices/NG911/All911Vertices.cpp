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
   dispNum_ = nullptr; 
   respNum_ = nullptr;
}

All911Vertices::~All911Vertices() {
   if (size_ != 0) {
      delete[] callNum_;
      delete[] dispNum_;
      delete[] respNum_;
   }

   callNum_ = nullptr;
   dispNum_ = nullptr;
   respNum_ = nullptr;
}

// Allocate memory for all class properties
void All911Vertices::setupVertices() {
   AllVertices::setupVertices();

   callNum_ = new int[size_];
   dispNum_ = new int[size_];
   respNum_ = new int[size_];

   // Populate arrays with 0
   fill_n(callNum_, size_, 0);
   fill_n(dispNum_, size_, 0);
   fill_n(respNum_, size_, 0);

   edgeMap_ = Simulator::getInstance().getModel()->getConnections()->getEdgeIndexMap();
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

   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {  
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

   // Create all psaps
   // Dispatchers in a psap = [callers in the zone * k] + some randomness
   for (int i = 0; i < psapList.size(); i++) {
      int psapQ = layout911->zone(i);
      int dispCount = (callersPerZone[psapQ] * dispNumScale_) + rng.inRange(-5, 5);
      if (dispCount < 1) { dispCount = 1; }
      dispNum_[psapList[i]] = dispCount;
   }

   // Create all responders
   // Responders in a node = [callers in the zone * k]/[number of responder nodes] + some randomness
   for (int i = 0; i < respList.size(); i++) {
      int respQ = layout911->zone(respList[i]);
      int respCount = (callersPerZone[respQ] * respNumScale_)/respPerZone[respQ] + rng.inRange(-5, 5);
      if (respCount < 1) { respCount = 1; }
      respNum_[respList[i]] = respCount;
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
         case RESP:   advanceRESP(i, layout);
            break;
         case CALR:   advanceCALR(i);
            break;
         default:
            break;
      }
   }

   //       // notify the source and destination edges if anything has happened to the vertex
   //       if (hasFired_[idx]) {
   //          LOG4CPLUS_DEBUG(vertexLogger_, "Vertex: " << idx << " has fired at time: "
   //                         << g_simulationStep * Simulator::getInstance().getDeltaT());
   //          // notify outgoing edges
   //          BGSIZE edgeCounts;
   //          if (edgeIndexMap != nullptr) {
   //             edgeCounts = edgeIndexMap->outgoingEdgeCount_[idx];
   //             if (edgeCounts != 0) {
   //                int beginIndex = edgeIndexMap->outgoingEdgeBegin_[idx];
   //                BGSIZE iEdg;
   //                for (BGSIZE i = 0; i < edgeCounts; i++) {
   //                   iEdg = edgeIndexMap->outgoingEdgeBegin_[beginIndex + i];
   //                   allEdges.preSpikeHit(iEdg);
   //                }
   //             }
   //          }
   //          // notify incoming edges
   //          edgeCounts = allEdges.edgeCounts_[idx];
   //          BGSIZE synapse_notified = 0;
   //          hasFired_[idx] = false;
   //       }
   //    }
}

void All911Vertices::advancePSAP(const int index) {
   /*
   // 911TODO
   */
}

void All911Vertices::advanceCALR(const int index) {
   /*
   // 911TODO
   if (busy[index]) {
      // if it's received a spike hit:
         // record time
         // refill RESP
         busy[index] = false;
   } else {
      if (rng.rand() > threshold_) {
         noise[index] = dispNum_[index];
         busy[index] = true;
         // record time

         // Notify edges?
      }
   }
   */
}

void All911Vertices::advanceRESP(const int index, shared_ptr<Layout> layout) {
   /*
   // 911TODO
   // if it's got a hit
      // reduce RESP resource
      int penalty = round((*layout->dist_)(index, 4));
      // Notify edges
   */
}
#endif