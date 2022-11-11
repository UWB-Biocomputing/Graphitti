/**
 * @file GraphManager.h
 *
 * @brief A class that exposes parts of the Boost Graph Library (BGL) needed for
 * loading a graphml file used to set up the initial simulation graph.
 *
 * @ingroup Simulator/Utils
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

class GraphManager
{
public:
    /// typedef for graphml graph type (adjacency list)
    typedef boost::adjacency_list<boost::vecS, boost::vecS,
            boost::directedS, VertexProperty, EdgeProperty,
            GraphProperty> Graph;

    typedef typename boost::graph_traits<Graph>::edge_iterator EdgeIterator;
    typedef typename boost::graph_traits<Graph>::vertex_iterator VertexIterator;
    typedef typename boost::graph_traits<Graph>::edge_descriptor EdgeDescriptor;

    static GraphManager &getInstance()
    {
        static GraphManager instance;
        return instance;
    }

    void setFilePath(string filePath)
    {
        graphFilePath_ = filePath;
    }

    /// @brief Registers a graph property with its attribute name in the graphml file
    ///
    /// Note: We are passing a pointer to a data member of the Struct that will hold
    /// the property. The BGL will use this when loading the graphML file.
    /// Reference: https://www.studytonight.com/cpp/pointer-to-members.php
    ///
    /// @tparam Property    Pointer to a Struct data member that will hold the property
    /// @param propName     The attribute name inthe graphml file
    /// @param property     Pointer to the property to be registered
    template <class Property>
    inline void registerProperty(const string &propName, Property property)
    {
        dp_.property(propName, boost::get(property, graph_));
    }

    /// @brief Loads a graph from a graphml file into a BGL graph
    /// @return The graph loaded as an adjacency list
    bool readGraph()
    {
        // Load graphml file into a BGL graph
        ifstream graph_file;

        // string file_name;
        string path = "//graphmlFile/text()";
        if (!ParameterManager::getInstance().getStringByXpath(path,
                                                              graphFilePath_)) {
            cerr << "Could not find XML path: " << path << ".\n";
            return false;
        };

        graph_file.open(graphFilePath_.c_str());
        if (!graph_file.is_open()) {
            cerr << "Failed to open file: " << graphFilePath_ << ".\n";
            return false;
        }

        boost::read_graphml(graph_file, graph_, dp_);
        return true;
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

    /// @brief Retrieves the number of vertices in the current graph
    /// @return The number of vertices in the current graph
    size_t numVertices() const
    {
        return boost::num_vertices(graph_);
    }

    /// @brief Retrieves the number of edges in the current graph
    /// @return The number of edges in the current graph
    size_t numEdges() const
    {
        return boost::num_edges(graph_);
    }

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
