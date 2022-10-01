/**
* @file Layout911.cpp
* 
* @ingroup Simulator/Layouts/NG911
*
* @brief The Layout class defines the layout of vertices in networks
*/

// #include <string>
#include "Layout911.h"
#include "ParameterManager.h"

Layout911::Layout911()
{
}

Layout911::~Layout911()
{
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

   // We must register the graph properties before loading it
   gm_.registerProperty("id", &VertexProperty::id);
   gm_.registerProperty("type", &VertexProperty::type);
   gm_.loadGraph();
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
   map<string, vertexType> vTypeMap = {{"CALR", vertexType::CALR},
                                       {"RESP", vertexType::RESP},
                                       {"PSAP", vertexType::PSAP}};
   map<string, int> vTypeCount;  // Count map for debugging

   // Add all vertices
   GraphManager<VertexProperty>::VertexIterator vi, vi_end;
   for (boost::tie(vi, vi_end) = gm_.vertices(); vi != vi_end; ++vi) {
      assert(*vi < numVertices);
      vertexTypeMap_[*vi] = vTypeMap[gm_[*vi].type];
      vTypeCount[gm_[*vi].type] += 1;
   }

   LOG4CPLUS_DEBUG(fileLogger_, "\nVERTEX TYPE MAP" << endl
                                                    << "\tTotal vertices: " << numVertices << endl
                                                    << "\tCaller vertices: " << vTypeCount["CALR"] << endl
                                                    << "\tPSAP vertices: " << vTypeCount["PSAP"] << endl
                                                    << "\tResponder vertices: " << vTypeCount["RESP"]
                                                    << endl);

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
   else if (vertexTypeMap_[srcVertex] == RESP && vertexTypeMap_[destVertex] == CALR)
      return RC;
   else if (vertexTypeMap_[srcVertex] == PSAP && vertexTypeMap_[destVertex] == PSAP)
      return PP;

   return ETYPE_UNDEF;
}

GraphManager<Layout911::VertexProperty> Layout911::getGraphManager()
{
   return gm_;
}