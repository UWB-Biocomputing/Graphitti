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
   GraphManager<NG911VertexProperties> &gm = GraphManager<NG911VertexProperties>::getInstance();
   gm.registerProperty("objectID", &NG911VertexProperties::objectID);
   gm.registerProperty("name", &NG911VertexProperties::name);
   gm.registerProperty("type", &NG911VertexProperties::type);
   gm.registerProperty("y", &NG911VertexProperties::y);
   gm.registerProperty("x", &NG911VertexProperties::x);
   gm.registerProperty("servers", &NG911VertexProperties::servers);
   gm.registerProperty("trunks", &NG911VertexProperties::trunks);
   gm.registerProperty("segments", &NG911VertexProperties::segments);
}

// Loads Layout911 member variables.
void Layout911::loadParameters()
{
   // Get the number of verticese from the GraphManager
   numVertices_ = GraphManager<NG911VertexProperties>::getInstance().numVertices();
}

// Setup the internal structure of the class.
void Layout911::setup()
{
   // Base class allocates memory for: xLoc_, yLoc, dist2_, and dist_
   // so we call its method first
   Layout::setup();

   // Loop over all vertices and set their x and y locations
   GraphManager<NG911VertexProperties>::VertexIterator vi, vi_end;
   GraphManager<NG911VertexProperties> &gm = GraphManager<NG911VertexProperties>::getInstance();
   for (boost::tie(vi, vi_end) = gm.vertices(); vi != vi_end; ++vi) {
      assert(*vi < numVertices_);
      xloc_[*vi] = gm[*vi].x;
      yloc_[*vi] = gm[*vi].y;
   }

   // Now we cache the between each pair of vertices distances^2 into a matrix
   for (int n = 0; n < numVertices_ - 1; n++) {
      for (int n2 = n + 1; n2 < numVertices_; n2++) {
         // distance^2 between two points in point-slope form
         dist2_(n, n2) = (xloc_[n] - xloc_[n2]) * (xloc_[n] - xloc_[n2])
                         + (yloc_[n] - yloc_[n2]) * (yloc_[n] - yloc_[n2]);

         // both points are equidistant from each other
         dist2_(n2, n) = dist2_(n, n2);
      }
   }

   // Finally take the square root to get the distances
   dist_ = sqrt(dist2_);
}

// Prints out all parameters to logging file.
void Layout911::printParameters() const
{
}

// Creates a vertex type map.
void Layout911::generateVertexTypeMap()
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
   GraphManager<NG911VertexProperties>::VertexIterator vi, vi_end;
   GraphManager<NG911VertexProperties> &gm = GraphManager<NG911VertexProperties>::getInstance();
   LOG4CPLUS_DEBUG(fileLogger_, "\nvertices in graph: " << gm.numVertices());
   for (boost::tie(vi, vi_end) = gm.vertices(); vi != vi_end; ++vi) {
      assert(*vi < numVertices_);
      vertexTypeMap_[*vi] = vTypeMap[gm[*vi].type];
      vTypeCount[gm[*vi].type] += 1;
   }

   // Register vertexTypes with recorder
   Recorder &recorder = Simulator::getInstance().getModel().getRecorder();
   recorder.registerVariable("vertexTypeMap", vertexTypeMap_, Recorder::UpdatedType::CONSTANT);

   LOG4CPLUS_DEBUG(fileLogger_, "\nVERTEX TYPE MAP"
                                   << endl
                                   << "\tTotal vertices: " << numVertices_ << endl
                                   << "\tCaller vertices: " << vTypeCount["CALR"] << endl
                                   << "\tPSAP vertices: " << vTypeCount["PSAP"] << endl
                                   << "\tLaw vertices: " << vTypeCount["LAW"] << endl
                                   << "\tFire vertices: " << vTypeCount["FIRE"] << endl
                                   << "\tEMS vertices: " << vTypeCount["EMS"] << endl);

   LOG4CPLUS_INFO(fileLogger_, "Finished initializing vertex type map");
}

// Returns the type of synapse at the given coordinates
edgeType Layout911::edgType(int srcVertex, int destVertex)
{
   if (vertexTypeMap_[srcVertex] == vertexType::CALR
       && vertexTypeMap_[destVertex] == vertexType::PSAP)
      return edgeType::CP;
   else if (vertexTypeMap_[srcVertex] == vertexType::PSAP
            && (vertexTypeMap_[destVertex] == vertexType::LAW
                || vertexTypeMap_[destVertex] == vertexType::FIRE
                || vertexTypeMap_[destVertex] == vertexType::EMS))
      return edgeType::PR;
   else if (vertexTypeMap_[srcVertex] == vertexType::PSAP
            && vertexTypeMap_[destVertex] == vertexType::CALR)
      return edgeType::PC;
   else if (vertexTypeMap_[srcVertex] == vertexType::PSAP
            && vertexTypeMap_[destVertex] == vertexType::PSAP)
      return edgeType::PP;
   else if ((vertexTypeMap_[srcVertex] == vertexType::LAW
             || vertexTypeMap_[destVertex] == vertexType::FIRE
             || vertexTypeMap_[destVertex] == vertexType::EMS)
            && vertexTypeMap_[destVertex] == vertexType::PSAP)
      return edgeType::RP;
   else if ((vertexTypeMap_[srcVertex] == vertexType::LAW
             || vertexTypeMap_[destVertex] == vertexType::FIRE
             || vertexTypeMap_[destVertex] == vertexType::EMS)
            && vertexTypeMap_[destVertex] == vertexType::CALR)
      return edgeType::RC;
   else
      return edgeType::ETYPE_UNDEF;
}


// Calculates the distance between the given vertex and the (x, y) coordinates of a point
double Layout911::getDistance(int vertexId, double x, double y)
{
   return sqrt(pow(x - xloc_[vertexId], 2) + (pow(y - yloc_[vertexId], 2)));
}
