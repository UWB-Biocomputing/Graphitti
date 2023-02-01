/**
* @file Layout911.cpp
* 
* @ingroup Simulator/Layouts/NG911
*
* @brief The Layout class defines the layout of vertices in networks
*/

// #include <string>
#include "Layout911.h"
#include "GraphManager.h"
#include "ParameterManager.h"

Layout911::Layout911()
{
}

Layout911::~Layout911()
{
}

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
   gm.registerProperty("type", &VertexProperty::type);
   gm.registerProperty("y", &VertexProperty::y);
   gm.registerProperty("x", &VertexProperty::x);
}

void Layout911::loadParameters()
{
   // Get the file paths for the vertex lists from the configuration file
   string callerFilePath;
   string psapFilePath;
   string responderFilePath;
   if (!ParameterManager::getInstance().getStringByXpath("//LayoutFiles/callersListFileName/text()",
                                                         callerFilePath)) {
      throw runtime_error("In Layout::loadParameters() caller "
                          "vertex list file path wasn't found and will not be initialized");
   }
   if (!ParameterManager::getInstance().getStringByXpath("//LayoutFiles/PSAPsListFileName/text()",
                                                         psapFilePath)) {
      throw runtime_error("In Layout::loadParameters() psap "
                          "vertex list file path wasn't found and will not be initialized");
   }
   if (!ParameterManager::getInstance().getStringByXpath(
          "//LayoutFiles/respondersListFileName/text()", responderFilePath)) {
      throw runtime_error("In Layout::loadParameters() responder"
                          "vertex list file path wasn't found and will not be initialized");
   }

   // Get the number of verticese from the GraphManager
   numVertices_ = GraphManager::getInstance().numVertices();
}

void Layout911::setup()
{
   // Base class allocates memory for: xLoc_, yLoc, dist2_, and dist_
   // so we call its method first
   Layout::setup();

   // Loop over all vertices and set thir x and y locations
   GraphManager::VertexIterator vi, vi_end;
   GraphManager &gm = GraphManager::getInstance();
   for (boost::tie(vi, vi_end) = gm.vertices(); vi != vi_end; ++vi) {
      assert(*vi < numVertices_);
      (*xloc_)[*vi] = gm[*vi].x;
      (*yloc_)[*vi] = gm[*vi].y;
   }

   // Now we cache the between each pair of vertices distances^2 into a matrix
   for (int n = 0; n < numVertices_ - 1; n++) {
      for (int n2 = n + 1; n2 < numVertices_; n2++) {
         // distance^2 between two points in point-slope form
         (*dist2_)(n, n2) = ((*xloc_)[n] - (*xloc_)[n2]) * ((*xloc_)[n] - (*xloc_)[n2])
                            + ((*yloc_)[n] - (*yloc_)[n2]) * ((*yloc_)[n] - (*yloc_)[n2]);

         // both points are equidistant from each other
         (*dist2_)(n2, n) = (*dist2_)(n, n2);
      }
   }

   // Finally take the square root to get the distances
   (*dist_) = sqrt((*dist2_));
}

void Layout911::printParameters() const
{
}

/// Creates a vertex type map.
/// @param  numVertices number of the vertices to have in the type map.
void Layout911::generateVertexTypeMap(int numVertices)
{
   DEBUG(cout << "\nInitializing vertex type map" << endl;);

   // Map vertex type string to vertexType
   // In the GraphML file Responders are divided in LAW, FIRE, and EMS.
   // Perhaps, we need to expand the vertex types?
   map<string, vertexType> vTypeMap = {{"CALR", vertexType::CALR},
                                       {"LAW", vertexType::RESP},
                                       {"FIRE", vertexType::RESP},
                                       {"EMS", vertexType::RESP},
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

void Layout911::initStarterMap(const int numVertices)
{
   Layout::initStarterMap(numVertices);
}

// Get the zone of the vertex
// Only built for 10x10 grid
// See: https://docs.google.com/spreadsheets/d/1DqP8sjkfJ_pkxtETzuEdoVZbWOGu633EMQAeShe5k68/edit?usp=sharing
int Layout911::zone(int index)
{
   return (index % 10 >= 5) + 2 * (index < 50);
}

///  Returns the type of synapse at the given coordinates
///
///  @param    srcVertex  integer that points to a Neuron in the type map as a source.
///  @param    destVertex integer that points to a Neuron in the type map as a destination.
///  @return type of the synapse.
edgeType Layout911::edgType(const int srcVertex, const int destVertex)
{
   if (vertexTypeMap_[srcVertex] == CALR && vertexTypeMap_[destVertex] == PSAP)
      return CP;
   else if (vertexTypeMap_[srcVertex] == PSAP && vertexTypeMap_[destVertex] == RESP)
      return PR;
   else if (vertexTypeMap_[srcVertex] == PSAP && vertexTypeMap_[destVertex] == CALR)
      return PC;
   else if (vertexTypeMap_[srcVertex] == PSAP && vertexTypeMap_[destVertex] == PSAP)
      return PP;
   else if (vertexTypeMap_[srcVertex] == RESP && vertexTypeMap_[destVertex] == PSAP)
      return RP;
   else
      return ETYPE_UNDEF;
}