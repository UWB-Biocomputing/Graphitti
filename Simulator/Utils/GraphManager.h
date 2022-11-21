/**
 * @file GraphManager.h
 * @author Jardi A. M. Jordan (jardiamj@gmail.com)
 * @date 11-11-2022
 * Supervised by Dr. Michael Stiber, UW Bothell CSSE Division
 * @ingroup Simulator/Utils
 * 
 * @brief This is a wrapper around the Boost Graph Library (BGL).
 * 
 * It is used to read graphml files that hold the initial simulation graph.
 *
 * The class provides a simple interface to load a graphML file. The BGL needs
 * to know the vertices, edges, graph properties before loading the graph. We
 * tell BGL where to store these properties and their type by registering them
 * before hand using the registerProperty() method.
 * Assumptions:
 *   - VertexProperty is a struct that contains the properties related to vertices
 *     lisgted in the graphml file.
 *   - EdgeProperty is a struct that contains the properties related to edges
 *     listed inthe graphml file.
 *   - GraphProperty is a structure that contains the properties related to the graph.
 *   - All relevant properties are registered via the `registerProperty()` method
 *     before calling `readGraph`.
 *   - Properties not registered are ignored.
 *
 * The structures for the VertexProperty, EdgeProperty, and GraphProperty are declared
 * in Global.h.
 * 
 * The class was made a Singleton because is needed in various places to initialize
 * the differen graph structures of the Simulator.
 *
 */

#pragma once

#include "Global.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphml.hpp>
#include <string>
#include <utility>

using namespace std;

class GraphManager {
public:
   /// typedef for graphml graph type (adjacency list)
   typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, VertexProperty,
                                 EdgeProperty, GraphProperty>
      Graph;

   typedef typename boost::graph_traits<Graph>::edge_iterator EdgeIterator;
   typedef typename boost::graph_traits<Graph>::vertex_iterator VertexIterator;
   typedef typename boost::graph_traits<Graph>::edge_descriptor EdgeDescriptor;

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

   /// @brief Direct access to the VertexProperty of a vertex descriptor
   /// @param vertex   the vertex descriptor (index)
   /// @return the VertexProperty of the vertex descriptor
   VertexProperty &operator[](size_t vertex);

   /// @brief Direct access to the VertexProperty of a vertex descriptor
   /// @param vertex   the vertex descriptor (index)
   /// @return the VertexProperty of the vertex descriptor
   const VertexProperty &operator[](size_t vertex) const;

   /// @brief Returns a list of EdgeDescriptors in ascending order by target vertexID
   /// @return List of EdgeDescriptors in ascending order by target vertexID
   const list<EdgeDescriptor> edgesSortByTarget() const;

   /// @brief Retrieves the number of vertices in the current graph
   /// @return The number of vertices in the current graph
   size_t numVertices() const;

   /// @brief Retrieves the number of edges in the current graph
   /// @return The number of edges in the current graph
   size_t numEdges() const;

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

   GraphManager(GraphManager const &) = delete;
   void operator=(GraphManager const &) = delete;
};
