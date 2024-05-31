/**
 * @file GraphManager.cpp
 * @author Jardi A. Martinez Jordan (jardiamj@gmail.com)
 * @date 11-11-2022
 * @ingroup Simulator/Utils
 * 
 * @brief The GraphManager is a wrapper around the Boost Graph Library (BGL).
 * 
 * It is used to read graphml files that hold the initial simulation graph.
 * 
 */

#include "GraphManager.h"
#include "ParameterManager.h"

void GraphManager::setFilePath(string filePath)
{
   graphFilePath_ = filePath;
}

bool GraphManager::readGraph()
{
   // Load graphml file into a BGL graph
   ifstream graph_file;

   // If graphFilePath_ isn't already defined, get it from ParameterManager
   if (graphFilePath_ == "") {
      // string file_name;
      string path = "//graphmlFile/text()";
      if (!ParameterManager::getInstance().getStringByXpath(path, graphFilePath_)) {
         cerr << "Could not find XML path: " << path << ".\n";
         return false;
      };
   }

   graph_file.open(graphFilePath_.c_str());
   if (!graph_file.is_open()) {
      cerr << "Failed to open file: " << graphFilePath_ << ".\n";
      return false;
   }

   boost::read_graphml(graph_file, graph_, dp_);
   return true;
}

pair<GraphManager::VertexIterator, GraphManager::VertexIterator> GraphManager::vertices()
{
   return boost::vertices(graph_);
}

pair<GraphManager::EdgeIterator, GraphManager::EdgeIterator> GraphManager::edges() const
{
   return boost::edges(graph_);
}

size_t GraphManager::source(const GraphManager::EdgeDescriptor &edge) const
{
   return boost::source(edge, graph_);
}

size_t GraphManager::target(const GraphManager::EdgeDescriptor &edge) const
{
   return boost::target(edge, graph_);
}

VertexProperty &GraphManager::operator[](size_t vertex)
{
   return graph_[vertex];
}

const VertexProperty &GraphManager::operator[](size_t vertex) const
{
   return graph_[vertex];
}

const list<GraphManager::EdgeDescriptor> GraphManager::edgesSortByTarget() const
{
   list<EdgeDescriptor> ei_list;
   EdgeIterator ei, ei_end;
   for (boost::tie(ei, ei_end) = edges(); ei != ei_end; ++ei) {
      ei_list.push_back(*ei);
   }

   // Use a lambda function for sorting the list of edges
   ei_list.sort([this](EdgeDescriptor const &a, EdgeDescriptor const &b) {
      return this->target(a) < this->target(b);
   });

   return ei_list;
}

size_t GraphManager::numVertices() const
{
   return boost::num_vertices(graph_);
}

size_t GraphManager::numEdges() const
{
   return boost::num_edges(graph_);
}
