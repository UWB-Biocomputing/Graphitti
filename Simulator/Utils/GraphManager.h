/**
 * @file GraphManager.h
 * @author Jardi A. M. Jordan (jardiamj@gmail.com)
 * @author Jasleen Kaur Saini (jasleenksaini@gmail.com)
 * @date 02-18-2025
 * Supervised by Dr. Michael Stiber, UW Bothell CSSE Division
 * @ingroup Simulator/Utils
 * 
 * @brief A templated wrapper around the Boost Graph Library (BGL).
 * 
 * The GraphManager class is responsible for reading and managing GraphML files 
 * that define the initial simulation graph structure. It provides a simple 
 * interface for loading graphs while ensuring that BGL correctly associates 
 * graph elements with their respective properties.
 * 
 * The templatized class is designed to support multiple applications by using  
 * `VertexProperties` struct that employs inheritance. This allows for specialization 
 * based on different simulation domains, such as:
 *   - Neuro: Graph structures used in Neural Network-specific simulations.
 *   - NG911: Graph structures used for Next Generation 911 emergency simulations.
 *
 * ## Assumptions:
 *   - `VertexProperties` is a base struct that includes application-specific properties 
 *     via inheritance. Derived structs are NG911Property and NeuralProperty.
 *   - `EdgeProperty` is a struct containing properties related to edges in the graph.
 *   - `GraphProperty` is a struct containing properties related to the entire graph.
 *   - All relevant properties are registered using the `registerProperty()` method 
 *     before calling `readGraph()`.
 *   - Properties not  registered will be ignored.
 *   - The entire GraphManager class is included in the header file to ensure that  
 *     the templated class can be compiled without requiring separate declarations.  
 *
 * The structures for `VertexProperties`, `EdgeProperties`, and `GraphProperties` 
 * are declared in `Global.h`.
 * 
 * This class follows the Singleton design pattern, ensuring a single instance 
 * is used throughout the simulation for consistent graph management.
 */

#pragma once

#include "Global.h"
#include "ParameterManager.h"
#include "Simulator.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphml.hpp>
#include <string>
#include <utility>

using namespace std;

template <typename VertexProperties> class GraphManager {
public:
   /// Using directive for graphml graph type (adjacency list)
   using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, VertexProperties,
                                       NeuralEdgeProperties, GraphProperties>;

   using EdgeIterator = typename boost::graph_traits<Graph>::edge_iterator;
   using VertexIterator = typename boost::graph_traits<Graph>::vertex_iterator;
   using EdgeDescriptor = typename boost::graph_traits<Graph>::edge_descriptor;

   /// @brief Returns a single instance of the GraphManager
   /// @return The single instance of the GraphManager
   static GraphManager &getInstance()
   {
      static GraphManager instance;
      return instance;
   }

   /// @brief Set the path for the graphML File to be read
   /// @param filePath The absolute path to the graphML file
   void setFilePath(string filePath);

   /// @brief Registers a graph property with its attribute name in the graphml file
   ///
   /// Note: We are passing a pointer to a data member of the Struct that will hold
   /// the property. The BGL will use this when loading the graphML file.
   /// Reference: https://www.studytonight.com/cpp/pointer-to-members.php
   ///
   /// @tparam Property    Pointer to a Struct data member that will hold the property
   /// @param propName     The attribute name inthe graphml file
   /// @param property     Pointer to the property to be registered
   template <class Property> inline void registerProperty(const string &propName, Property property)
   {
      dp_.property(propName, boost::get(property, graph_));
   }

   /// @brief Loads a graph from a graphml file into a BGL graph
   /// @return The graph loaded as an adjacency list
   bool readGraph();

   /// @brief Exposes the BGL Vertex Iterators of the stored Graph
   /// @return a pair of VertexIterators where first points to the beginning
   ///         and second points to the end of the vertices vector
   pair<VertexIterator, VertexIterator> vertices();

   /// @brief Exposes the BGL Edge Iterators of the stored Graph
   /// @return a pair of EdgeIterators where first points to the beginning
   ///         and second points to the end of the edges vector
   pair<EdgeIterator, EdgeIterator> edges() const;


   /// @brief Retrieves the source vertex index for the given Edge
   /// @param edge the EdgeDescriptor
   /// @return the source vertex index for the given Edge
   size_t source(const EdgeDescriptor &edge) const;

   /// @brief Retrieves the target vertex index for the given Edge
   /// @param edge the EdgeDescriptor
   /// @return the target vertex index for the given Edge
   size_t target(const EdgeDescriptor &edge) const;

   /// @brief Retrieves the weight of an edge
   /// @param edge the EdgeDescriptor
   /// @return the weight of the given edge
   double weight(const EdgeDescriptor &edge) const;

   /// @brief Direct access to the VertexProperties of a vertex descriptor
   /// @param vertex   the vertex descriptor (index)
   /// @return the VertexProperties of the vertex descriptor
   VertexProperties &operator[](size_t vertex);

   /// @brief Direct access to the VertexProperties of a vertex descriptor
   /// @param vertex   the vertex descriptor (index)
   /// @return the VertexProperties of the vertex descriptor
   const VertexProperties &operator[](size_t vertex) const;

   /// @brief Returns a list of EdgeDescriptors in ascending order by target vertexID
   /// @return List of EdgeDescriptors in ascending order by target vertexID
   const list<EdgeDescriptor> edgesSortByTarget() const;

   /// @brief Retrieves the number of vertices in the current graph
   /// @return The number of vertices in the current graph
   size_t numVertices() const;

   /// @brief Retrieves the number of edges in the current graph
   /// @return The number of edges in the current graph
   size_t numEdges() const;

   /// Delete copy and move methods to avoid copy instances of the singleton
   GraphManager(const GraphManager &graphManager) = delete;
   GraphManager &operator=(const GraphManager &graphManager) = delete;

   GraphManager(GraphManager &&graphManager) = delete;
   GraphManager &operator=(GraphManager &&graphManager) = delete;

private:
   /// stores the graph
   Graph graph_;
   string graphFilePath_;

   /// container for dynamic properties map
   boost::dynamic_properties dp_;

   /// @brief Constructor
   GraphManager() : graph_(), dp_(boost::ignore_other_properties)
   {
   }
};

/**
  * @class GraphManager
  * @brief A templated wrapper around the Boost Graph Library (BGL).
  */


/// @brief Sets the file path for the graphML file.
/// @param filePath The absolute path to the graphML file.
template <typename VertexProperties>
void GraphManager<VertexProperties>::setFilePath(string filePath)
{
   graphFilePath_ = filePath;
}

/// @brief Reads a graph from a GraphML file into a BGL graph.
/// @return True if the graph was successfully read, false otherwise.
template <typename VertexProperties> bool GraphManager<VertexProperties>::readGraph()
{
   // Load graphml file into a BGL graph
   ifstream graph_file;
   log4cplus::Logger consoleLogger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("console"));

   // If graphFilePath_ isn't already defined, get it from ParameterManager
   if (graphFilePath_ == "") {
      string path = "//graphmlFile/text()";
      if (!ParameterManager::getInstance().getStringByXpath(path, graphFilePath_)) {
         LOG4CPLUS_ERROR(consoleLogger,  ("Could not find XML path: " + path + ".\n"));
         return false;
      };
   }

   graph_file.open(graphFilePath_.c_str());
   if (!graph_file.is_open()) {
      LOG4CPLUS_ERROR(consoleLogger,  ("Failed to open file: " + graphFilePath_ + ".\n"));
      return false;
   }

   boost::read_graphml(graph_file, graph_, dp_);
   return true;
}

/// @brief Retrieves the vertices of the graph.
/// @return A pair of VertexIterators for the graph vertices.
template <typename VertexProperties>
pair<typename GraphManager<VertexProperties>::VertexIterator,
     typename GraphManager<VertexProperties>::VertexIterator>
   GraphManager<VertexProperties>::vertices()
{
   return boost::vertices(graph_);
}

/// @brief Retrieves the edges of the graph.
/// @return A pair of EdgeIterators for the graph edges.
template <typename VertexProperties>
pair<typename GraphManager<VertexProperties>::EdgeIterator,
     typename GraphManager<VertexProperties>::EdgeIterator>
   GraphManager<VertexProperties>::edges() const
{
   return boost::edges(graph_);
}

/// @brief Retrieves the source vertex of a given edge.
/// @param edge The edge descriptor.
/// @return The source vertex index.
template <typename VertexProperties>
size_t GraphManager<VertexProperties>::source(
   const typename GraphManager<VertexProperties>::EdgeDescriptor &edge) const
{
   return boost::source(edge, graph_);
}

/// @brief Retrieves the target vertex of a given edge.
/// @param edge The edge descriptor.
/// @return The target vertex index.
template <typename VertexProperties>
size_t GraphManager<VertexProperties>::target(
   const typename GraphManager<VertexProperties>::EdgeDescriptor &edge) const
{
   return boost::target(edge, graph_);
}

/// @brief Retrieves the weight of an edge
/// @param edge the EdgeDescriptor
/// @return the weight of the given edge
template <typename VertexProperties>
double GraphManager<VertexProperties>::weight(
   const typename GraphManager<VertexProperties>::EdgeDescriptor &edge) const
{
   return boost::get(&NeuralEdgeProperties::weight, graph_, edge);
}

/// @brief Directly access the VertexProperties of a vertex descriptor.
/// @param vertex The vertex descriptor (index).
/// @return The VertexProperties of the vertex.
template <typename VertexProperties>
VertexProperties &GraphManager<VertexProperties>::operator[](size_t vertex)
{
   return graph_[vertex];
}

/// @brief Directly access the VertexProperties of a vertex descriptor (const).
/// @param vertex The vertex descriptor (index).
/// @return The VertexProperties of the vertex.
template <typename VertexProperties>
const VertexProperties &GraphManager<VertexProperties>::operator[](size_t vertex) const
{
   return graph_[vertex];
}

/// @brief Returns a list of EdgeDescriptors sorted by target vertexID.
/// @return A sorted list of EdgeDescriptors.
template <typename VertexProperties>
const list<typename GraphManager<VertexProperties>::EdgeDescriptor>
   GraphManager<VertexProperties>::edgesSortByTarget() const
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

/// @brief Retrieves the number of vertices in the current graph.
/// @return The number of vertices.
template <typename VertexProperties> size_t GraphManager<VertexProperties>::numVertices() const
{
   return boost::num_vertices(graph_);
}

/// @brief Retrieves the number of edges in the current graph.
/// @return The number of edges.
template <typename VertexProperties> size_t GraphManager<VertexProperties>::numEdges() const
{
   return boost::num_edges(graph_);
}
