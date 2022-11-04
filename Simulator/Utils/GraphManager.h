/**
 * @file GraphManager.h
 *
 * @brief A class that exposes parts of the Boost Graph Library (BGL) needed for
 * loading a graphml file used to set up the initial simulation graph.
 *
 * @ingroup Simulator/Utils
 *
 * The class provides a simple interface to load a 
 * graphml file with the following assumptions:
 *   - VertexProperty is a struct that contains the properties related to vertices
 *     lisgted in the graphml file.
 *   - EdgeProperty is a struct that contains the properties related to edges
 *     listed inthe graphml file.
 *   - GraphProperty is a structu that contains the properties related to the graph.
 *   - All relevant properties are registered via the `registerProperty()` method
 *     before calling `loadGraph`.
 *   - Properties not registered are ignored.
 *
 *
 * @author Jardi A. M. Jordan
 * Supervised by Dr. Michael Stiber, UW Bothell CSSE Division
 */

#pragma once

#include "ParameterManager.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphml.hpp>
#include <utility>
#include <string>

using namespace std;

template <class VertexProperty = boost::no_property, 
          class EdgeProperty = boost::no_property,
          class GraphProperty = boost::no_property>
class GraphManager
{
public:
    /// typedef for graphml graph type (adjacency list)
    typedef boost::adjacency_list<boost::vecS, boost::vecS,
            boost::undirectedS, VertexProperty, EdgeProperty,
            GraphProperty> Graph;

    typedef typename boost::graph_traits<Graph>::edge_iterator EdgeIterator;
    typedef typename boost::graph_traits<Graph>::vertex_iterator VertexIterator;
    typedef typename boost::graph_traits<Graph>::edge_descriptor EdgeDescriptor;

    /// @brief Constructor
    GraphManager() : graph_(), dp_(boost::ignore_other_properties) {}

    /// @brief Registers a graph property with its attribute name in the graphml file
    /// @tparam Property    Type of the property to be registered
    /// @param propName     The attribute name inthe graphml file
    /// @param property     Pointer to the property to be registered
    template <class Property>
    inline void registerProperty(const string &propName, Property property)
    {
        dp_.property(propName, boost::get(property, graph_));
    }

    /// @brief Loads a graph from a graphml file into a BGL graph
    /// @return The graph loaded as an adjacency list
    Graph &loadGraph()
    {
        // Load graphml file into a BGL graph
        ifstream graph_file;
        if (!ParameterManager::getInstance().getFileByXpath("//graphmlFile/text()",
                                                            graph_file)) {
            throw runtime_error("In GraphManager::loadGraph() "
                                "graphml file wasn't found and won't be initialized");
        };
        boost::read_graphml(graph_file, graph_, dp_);
        return graph_;
    }

    /// @brief Exposes the BGL Vertex Iterators of the stored Graph
    /// @return a pair of VertexIterators where first points to the beginning
    ///         and second points to the end of the vertices vector
    pair<VertexIterator, VertexIterator> vertices()
    {
        return boost::vertices(graph_);
    }

    /// @brief Exposes the BGL Edge Iterators of the stored Graph
    /// @return a pair of EdgeIterators where first points to the beginning
    ///         and second points to the end of the edges vector
    pair<EdgeIterator, EdgeIterator> edges() const
    {
        return boost::edges(graph_);
    }

    /// @brief Retrieves the source vertex index for the given Edge
    /// @param edge the EdgeDescriptor
    /// @return the source vertex index for the given Edge
    size_t source(const EdgeDescriptor &edge) const
    {
        return boost::source(edge, graph_);
    }

    /// @brief Retrieves the target vertex index for the given Edge
    /// @param edge the EdgeDescriptor
    /// @return the target vertex index for the given Edge
    size_t target(const EdgeDescriptor &edge) const
    {
        return boost::target(edge, graph_);
    }

    /// @brief Direct access to the VertexProperty of a vertex descriptor
    /// @param vertex   the vertex descriptor (index)
    /// @return the VertexProperty of the vertex descriptor
    VertexProperty &operator[](size_t vertex)
    {
        return graph_[vertex];
    }

    /// @brief Direct access to the VertexProperty of a vertex descriptor
    /// @param vertex   the vertex descriptor (index)
    /// @return the VertexProperty of the vertex descriptor
    const VertexProperty &operator[](size_t vertex) const
    {
        return graph_[vertex];
    }

    /// @brief Returns a list of EdgeDescriptors in ascending order by target vertexID
    /// @return List of EdgeDescriptors in ascending order by target vertexID
    const list<EdgeDescriptor> edgesSortByTarget() const
    {
        list<EdgeDescriptor> ei_list;
        EdgeIterator ei, ei_end;
        for (boost::tie(ei, ei_end) = edges(); ei != ei_end; ++ei) {
            ei_list.push_back(*ei);
        }

        ei_list.sort([this](EdgeDescriptor const& a, EdgeDescriptor const& b){
            return this->target(a) < this->target(b);
        });

        return ei_list;
    }

private:
    /// stores the graph
    Graph graph_;

    /// container for dynamic properties map
    boost::dynamic_properties dp_;
};
