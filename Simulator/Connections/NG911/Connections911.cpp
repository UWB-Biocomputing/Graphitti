/**
* @file Connections911.cpp
*
* @ingroup Simulator/Connections/NG911
* 
* @brief The model of the static network
*
*/

#include "Connections911.h"
#include "ParameterManager.h"
#include "All911Vertices.h"

Connections911::Connections911() {

}

Connections911::~Connections911() {

}

void Connections911::setupConnections(Layout *layout, IAllVertices *vertices, AllEdges *edges) {
   int numVertices = Simulator::getInstance().getTotalVertices();

   int added = 0;

   LOG4CPLUS_INFO(fileLogger_, "Initializing connections");

   // For each source vertex
   for (int srcVertex = 0; srcVertex < numVertices; srcVertex++) {
      int connsAdded = 0;

      // For each destination verex
      for (int destVertex = 0; destVertex < numVertices; destVertex++) {
         if (connsAdded >= connsPerVertex_) { break; }
         if (srcVertex == destVertex) { continue; }

         BGFLOAT dist = (*layout->dist_)(srcVertex, destVertex);
         edgeType type = layout->edgType(srcVertex, destVertex);

         // Undefined edge types
         if (type == ETYPE_UNDEF) { continue; }

         // Zone each vertex belongs to
         int srcZone = layout->zone(srcVertex);
         int destZone = layout->zone(destVertex);

         // CP and PR where they aren't in the same zone
         // All PP and RC are defined
         if (type == CP || type == PR) {
            if (srcZone != destZone) { continue; }
         }

         BGFLOAT *sumPoint = &(dynamic_cast<AllVertices *>(vertices)->summationMap_[destVertex]);

         LOG4CPLUS_DEBUG(fileLogger_, "Source: " << srcVertex << " Dest: " << destVertex << " Dist: "
                                       << dist);

         BGSIZE iEdg;
         edges->addEdge(iEdg, type, srcVertex, destVertex, sumPoint, Simulator::getInstance().getDeltaT());
         added++;
         connsAdded++;
      }
   }

   LOG4CPLUS_DEBUG(fileLogger_,"Added connections: " << added);
}

void Connections911::loadParameters() {
   ParameterManager::getInstance().getIntByXpath("//connsPerVertex/text()", connsPerVertex_);
   ParameterManager::getInstance().getIntByXpath("//psapsToErase/text()", psapsToErase_);
   ParameterManager::getInstance().getIntByXpath("//respsToErase/text()", respsToErase_);
}

void Connections911::printParameters() const {

}

///  Update the connections status in every epoch.
///
///  @param  vertices  The Vertex list to search from.
///  @param  layout   Layout information of the vertex network.
///  @return true if successful, false otherwise.
bool Connections911::updateConnections(IAllVertices &vertices, Layout *layout) {
   // Only run on the first epoch
   if (Simulator::getInstance().getCurrentStep() != 1) { return false; }

   for (int i = 0; i < psapsToErase_; i++) {
      erasePSAP(vertices, layout);
   }

   for (int i = 0; i < respsToErase_; i++) {
      eraseRESP(vertices, layout);
   }

   return true;
}

///  Randomly delete 1 PSAP and rewire all the edges around it.
///
///  @param  vertices  The Vertex list to search from.
///  @param  layout   Layout information of the vertex network.
///  @return true if successful, false otherwise.
bool Connections911::erasePSAP(IAllVertices &vertices, Layout *layout) {
   int numVertices = Simulator::getInstance().getTotalVertices();

   vector<int> psaps;
   psaps.clear();

   // Find all psaps
   for (int i = 0; i < numVertices; i++) {
      if (layout->vertexTypeMap_[i] == PSAP) {
         psaps.push_back(i);
      }
   }

   // Only 1 psap, do not delete me :(
   if (psaps.size() < 2) { return false; }

   // Pick random PSAP
   int randVal = rng.inRange(0, psaps.size());
   int randPSAP = psaps[randVal];
   psaps.erase(psaps.begin() + randVal);

   BGSIZE maxTotalEdges = edges_->maxEdgesPerVertex_ * numVertices;
   bool changesMade = false;
   vector<int> callersToReroute;
   vector<int> respsToReroute;

   // Iterate through all edges
   for (int iEdg = 0; iEdg < maxTotalEdges; iEdg++) {
      if (!edges_->inUse_[iEdg]) { continue; }
      int srcVertex = edges_->sourceVertexIndex_[iEdg];
      int dstVertex = edges_->destVertexIndex_[iEdg];

      // Find PSAP edge
      if (srcVertex == randPSAP || dstVertex == randPSAP) {
         changesMade = true;
         edges_->eraseEdge(dstVertex, iEdg);

         // This is here so that we don't delete the vertex if we can't find any edges
         layout->vertexTypeMap_[randPSAP] = VTYPE_UNDEF;

         // Identify all psap-less callers
         if (layout->vertexTypeMap_[srcVertex] == CALR) {
            callersToReroute.push_back(srcVertex);
         }

         // Identify all psap-less responders
         if (layout->vertexTypeMap_[dstVertex] == RESP) {
            respsToReroute.push_back(dstVertex);
         }
      }
   }

   // Failsafe
   if (psaps.size() < 1) { return false; }

   // For each psap-less caller, find closest match
   for (int i = 0; i < callersToReroute.size(); i++) {
      int srcVertex = callersToReroute[i];

      int closestPSAP = psaps[0];
      BGFLOAT smallestDist = (*layout->dist_)(srcVertex, closestPSAP);

      // Find closest PSAP
      for (int i = 0; i < psaps.size(); i++) {
         BGFLOAT dist = (*layout->dist_)(srcVertex, psaps[i]);
         if (dist < smallestDist) {
            smallestDist = dist;
            closestPSAP = psaps[i];
         }
      }

      // Insert Caller to PSAP edge
      BGFLOAT *sumPoint = &(dynamic_cast<AllVertices *>(&vertices)->summationMap_[closestPSAP]);
      BGSIZE iEdg;
      edges_->addEdge(iEdg, CP, srcVertex, closestPSAP, sumPoint, Simulator::getInstance().getDeltaT());
   }

   // For each psap-less responder, find closest match
   for (int i = 0; i < respsToReroute.size(); i++) {
      int dstVertex = respsToReroute[i];

      int closestPSAP = psaps[0];
      BGFLOAT smallestDist = (*layout->dist_)(closestPSAP, dstVertex);

      // Find closest PSAP
      for (int i = 0; i < psaps.size(); i++) {
         BGFLOAT dist = (*layout->dist_)(psaps[i], dstVertex);
         if (dist < smallestDist) {
            smallestDist = dist;
            closestPSAP = psaps[i];
         }
      }

      // Insert PSAP to Responder edge
      BGFLOAT *sumPoint = &(dynamic_cast<AllVertices *>(&vertices)->summationMap_[dstVertex]);
      BGSIZE iEdg;
      edges_->addEdge(iEdg, PR, closestPSAP, dstVertex, sumPoint, Simulator::getInstance().getDeltaT());
   }

   return changesMade;
}

///  Randomly delete 1 RESP.
///
///  @param  vertices  The Vertex list to search from.
///  @param  layout   Layout information of the vertex network.
///  @return true if successful, false otherwise.
bool Connections911::eraseRESP(IAllVertices &vertices, Layout *layout) {
   int numVertices = Simulator::getInstance().getTotalVertices();

   vector<int> resps;
   resps.clear();

   // Find all resps
   for (int i = 0; i < numVertices; i++) {
      if (layout->vertexTypeMap_[i] == RESP) {
         resps.push_back(i);
      }
   }

   // Only 1 resp, do not delete me :(
   if (resps.size() < 2) { return false; }

   // Pick random RESP
   int randVal = rng.inRange(0, resps.size());
   int randRESP = resps[randVal];
   resps.erase(resps.begin() + randVal);

   BGSIZE maxTotalEdges = edges_->maxEdgesPerVertex_ * numVertices;
   bool changesMade = false;

   // Iterate through all edges
   for (int iEdg = 0; iEdg < maxTotalEdges; iEdg++) {
      if (!edges_->inUse_[iEdg]) { continue; }
      int srcVertex = edges_->sourceVertexIndex_[iEdg];
      int dstVertex = edges_->destVertexIndex_[iEdg];

      // Find RESP edge
      if (srcVertex == randRESP || dstVertex == randRESP) {
         changesMade = true;
         edges_->eraseEdge(dstVertex, iEdg);

         // This is here so that we don't delete the vertex if we can't find any edges
         layout->vertexTypeMap_[randRESP] = VTYPE_UNDEF;
      }
   }

   return changesMade;
}