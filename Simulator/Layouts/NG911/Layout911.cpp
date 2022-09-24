/**
* @file Layout911.cpp
* 
* @ingroup Simulator/Layouts/NG911
*
* @brief The Layout class defines the layout of vertices in networks
*/

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

   // Initialize Vertex Lists based on the data read from the xml files
   if (!ParameterManager::getInstance().getIntVectorByXpath(callerFilePath, "C",
                                                            callerVertexList_)) {
      throw runtime_error("In Layout::loadParameters() "
                          "caller vertex list list file wasn't loaded correctly"
                          "\n\tfile path: "
                          + callerFilePath);
   }
   numCallerVertices_ = callerVertexList_.size();
   if (!ParameterManager::getInstance().getIntVectorByXpath(psapFilePath, "P", psapVertexList_)) {
      throw runtime_error("In Layout::loadParameters() "
                          "psap vertex list file wasn't loaded correctly."
                          "\n\tfile path: "
                          + psapFilePath);
   }
   if (!ParameterManager::getInstance().getIntVectorByXpath(responderFilePath, "R",
                                                            responderVertexList_)) {
      throw runtime_error("In Layout::loadParameters() "
                          "responder vertex list file wasn't loaded correctly."
                          "\n\tfile path: "
                          + responderFilePath);
   }
}

void Layout911::printParameters() const
{
}

/// Creates a vertex type map.
/// @param  numVertices number of the vertices to have in the type map.
void Layout911::generateVertexTypeMap(int numVertices)
{
   DEBUG(cout << "\nInitializing vertex type map" << endl;);

   Graph graph;
   boost::dynamic_properties dp(boost::ignore_other_properties);
   registerVertexProperties(dp, graph);

   // ToDo: ParameterManager could return the open graphml file
   string graph_file_name;
   if (!ParameterManager::getInstance().getStringByXpath("//graphmlFile/text()",
                                                         graph_file_name)) {
      throw runtime_error("In Connections911::setupConnections() "
                          "graphml file wasn't found and won't be initialized");
   };

   // Read graphml file
   ifstream graph_file(graph_file_name.c_str());
   if (!graph_file.is_open()) {
      throw runtime_error("In Connections911::setupConnections() "
                          "Loading graph file failed "
                          "\n\tfile path: " + graph_file_name);
   }

   boost::read_graphml(graph_file, graph, dp);

   map<string, vertexType> vTypeMap = {{"CALR", vertexType::CALR},
                                       {"RESP", vertexType::RESP},
                                       {"PSAP", vertexType::PSAP}};
   map<string, int> vTypeCount;

   // add all vertices
   boost::graph_traits<Graph>::vertex_iterator vi, vi_end;
   for (boost::tie(vi, vi_end) = boost::vertices(graph); vi != vi_end; ++vi) {
      assert(*vi < numVertices);
      vertexTypeMap_[*vi] = vTypeMap[graph[*vi].type];
      vTypeCount[graph[*vi].type] += 1;
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

void Layout911::registerVertexProperties(boost::dynamic_properties &dp, Graph &graph)
{
   dp.property("id", boost::get(&Layout911::VertexProperty::id, graph));
   dp.property("type", boost::get(&Layout911::VertexProperty::type, graph));
}