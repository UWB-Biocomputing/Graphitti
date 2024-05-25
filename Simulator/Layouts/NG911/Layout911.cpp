/**
* @file Layout911.cpp
* 
* @ingroup Simulator/Layouts/NG911
*
* @brief Specialization of the Layout class for the NG911 network
*/

#include "Layout911.h"
#include "GraphManager.h"
#include "ParameterManager.h"

// Register vertex properties with the GraphManager
void Layout911::registerGraphProperties()
{
   // The base class registers properties that are common to all vertices
   // TODO: Currently not implemented because Neuro model doesn't used graphML
   Layout::registerGraphProperties();

   // We must register the graph properties before loading it.
   // We are passing a pointer to a data member of the VertexProperty
   // so Boost Graph Library can use it for loading the graphML file.
   // Look at: https://www.studytonight.com/cpp/pointer-to-members.php
   GraphManager &gm = GraphManager::getInstance();
   gm.registerProperty("objectID", &VertexProperty::objectID);
   gm.registerProperty("name", &VertexProperty::name);
   gm.registerProperty("servers", &VertexProperty::servers);
   gm.registerProperty("trunks", &VertexProperty::trunks);
   gm.registerProperty("segments", &VertexProperty::segments);
}

// Loads Layout911 member variables.
void Layout911::loadParameters()
{
   // Get the number of verticese from the GraphManager
   numVertices_ = GraphManager::getInstance().numVertices();
}

// Setup the internal structure of the class.
void Layout911::setup() : Layout::setup()
{
   // Base class allocates memory for: xLoc_, yLoc, dist2_, and dist_
   // so we call its method first
}

// Prints out all parameters to logging file.
void Layout911::printParameters() const
{
}

// Creates a vertex type map.
void Layout911::generateVertexTypeMap(int numVertices)
{
   DEBUG(cout << "\nInitializing vertex type map" << endl;);

   // Map vertex type string to vertexType
   // In the GraphML file Responders are divided in LAW, FIRE, and EMS.
   // Perhaps, we need to expand the vertex types?
   map<string, vertexType> vTypeMap = {{"CALR", vertexType::CALR},
                                       {"LAW", vertexType::LAW},
                                       {"FIRE", vertexType::FIRE},
                                       {"EMS", vertexType::EMS},
                                       {"PSAP", vertexType::PSAP}};
   // Count map for debugging
   map<string, int> vTypeCount;

   // Add all vertices
   GraphManager::VertexIterator vi, vi_end;
   GraphManager &gm = GraphManager::getInstance();
   LOG4CPLUS_DEBUG(fileLogger_, "\nvertices in graph: " << gm.numVertices());
   for (boost::tie(vi, vi_end) = gm.vertices(); vi != vi_end; ++vi) {
      assert(*vi < numVertices_);
      vertexTypeMap_[*vi] = vTypeMap[gm[*vi].type];
      vTypeCount[gm[*vi].type] += 1;
   }

   LOG4CPLUS_DEBUG(fileLogger_, "\nVERTEX TYPE MAP"
                                   << endl
                                   << "\tTotal vertices: " << numVertices_ << endl
                                   << "\tCaller vertices: " << vTypeCount["CALR"] << endl
                                   << "\tPSAP vertices: " << vTypeCount["PSAP"] << endl
                                   << "\tResponder vertices: " << vTypeCount["RESP"] << endl);

   LOG4CPLUS_INFO(fileLogger_, "Finished initializing vertex type map");
}

// Returns the type of synapse at the given coordinates
edgeType Layout911::edgType(int srcVertex, int destVertex)
{
   if (vertexTypeMap_[srcVertex] == CALR && vertexTypeMap_[destVertex] == PSAP)
      return CP;
   else if (vertexTypeMap_[srcVertex] == PSAP
            && (vertexTypeMap_[destVertex] == LAW || vertexTypeMap_[destVertex] == FIRE
                || vertexTypeMap_[destVertex] == EMS))
      return PR;
   else if (vertexTypeMap_[srcVertex] == PSAP && vertexTypeMap_[destVertex] == CALR)
      return PC;
   else if (vertexTypeMap_[srcVertex] == PSAP && vertexTypeMap_[destVertex] == PSAP)
      return PP;
   else if ((vertexTypeMap_[srcVertex] == LAW || vertexTypeMap_[destVertex] == FIRE
             || vertexTypeMap_[destVertex] == EMS)
            && vertexTypeMap_[destVertex] == PSAP)
      return RP;
   else if ((vertexTypeMap_[srcVertex] == LAW || vertexTypeMap_[destVertex] == FIRE
             || vertexTypeMap_[destVertex] == EMS)
            && vertexTypeMap_[destVertex] == CALR)
      return RC;
   else
      return ETYPE_UNDEF;
}


// Calculates the distance between the given vertex and the (x, y) coordinates of a point
double Layout911::getDistance(int vertexId, double x, double y)
{
   return sqrt(pow(x - xloc_[vertexId], 2) + (pow(y - yloc_[vertexId], 2)));
}
