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
   vector<DistDestVertex> distDestVertices;
   vector<DistDestVertex> psapVertices;

   int added = 0;

   LOG4CPLUS_INFO(fileLogger_, "Initializing connections");

   // For each source vertex
   for (int srcVertex = 0; srcVertex < numVertices; srcVertex++) {
      distDestVertices.clear();
      psapVertices.clear();

      // For each destination verex
      for (int destVertex = 0; destVertex < numVertices; destVertex++) {
         if (srcVertex != destVertex) {
            BGFLOAT dist = (*layout->dist_)(srcVertex, destVertex);
            edgeType type = layout->edgType(srcVertex, destVertex);

            // If they are close enough and the edge type is defined
            if ((dist <= threshConnsRadius_ && type != ETYPE_UNDEF) || type == PP) {
               DistDestVertex distDestVertex;
               distDestVertex.dist = dist;
               distDestVertex.destVertex = destVertex;
               if (type != PP) {
                  distDestVertices.push_back(distDestVertex);
               }
               else {
                  psapVertices.push_back(distDestVertex);
               }
            }
         }
      }

      // Sort by distance to create the shortest ones first
      sort(distDestVertices.begin(), distDestVertices.end());
      distDestVertices.insert(distDestVertices.begin(), psapVertices.begin(), psapVertices.end());

      int uLimit = (connsPerVertex_ < distDestVertices.size()) ? connsPerVertex_ : distDestVertices.size();

      for (BGSIZE i = 0; i < distDestVertices.size() && (int) i < uLimit; i++) {
         int destVertex = distDestVertices[i].destVertex;
         edgeType type = layout->edgType(srcVertex, destVertex);
         BGFLOAT *sumPoint = &(dynamic_cast<AllVertices *>(vertices)->summationMap_[destVertex]);

         LOG4CPLUS_DEBUG(fileLogger_, "Source: " << srcVertex << " Dest: " << destVertex << " Dist: "
                                                 << distDestVertices[i].dist);

         BGSIZE iEdg;
         edges->addEdge(iEdg, type, srcVertex, destVertex, sumPoint, Simulator::getInstance().getDeltaT());
         added++;
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

