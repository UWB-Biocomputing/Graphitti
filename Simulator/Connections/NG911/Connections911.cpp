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

         // Quadrant each vertex belongs to
         int srcQuadrant = (srcVertex%10 >= 5) + 2*(srcVertex < 50);
         int destQuadrant = (destVertex%10 >= 5) + 2*(destVertex < 50);

         // CP and PR where they aren't in the same quadrant
         // All PP and RC are defined
         if (type == CP || type == PR) {
            if (srcQuadrant != destQuadrant) { continue; }
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
   ParameterManager::getInstance().getBGFloatByXpath("//threshConnsRadius/text()", threshConnsRadius_);
   ParameterManager::getInstance().getIntByXpath("//connsPerVertex/text()", connsPerVertex_);
}

void Connections911::printParameters() const {

}

