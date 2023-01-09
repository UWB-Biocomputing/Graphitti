## Initial Simulator System Graph

Graphitti being a graph based simulator, works with systems that can be expressed as a network of vertices (nodes) and edges (connections). The first models developed with Graphitti were Biological Neural Network ones, for many of these models the network starts as a set of disconnected neurons laid out in a grid structure that defines their X and Y locations. These neurons then form connections as the simulation goes on. Feeding the set of disconnected neurons to Graphitti was relatively simple, since all we needed was a list of identifiers. Neuron attributes, such as neuron type, were provided through separate files that contained a list of neuron identifiers for each attribute which worked well for neural network models. Unfortunately, it doesn't scale well for networks such as the NG911 with a predefined complex structure.

The edges between nodes in a NG911 network depends on their geographical location and service boundaries (or jurisdictional boundaries). We have decided that the best way to feed such a network to Graphitti is through a configuration file that contains the list of vertices and edges along with their attributes. We also chose to use a standardized file format, settling for [GraphML](http://graphml.graphdrawing.org/) because it is supported by the [Boost Graph Library](https://www.boost.org/doc/libs/1_81_0/libs/graph/doc/index.html), and other libaries and projects.

There are two main steps to implementing a graph representation of an NG911 network for Graphitti:

1. [Extracting the Graph from the entities geographical location and service boundaries](../Tools/GIStoGraph.md), and
2. Loading the Network from a GraphML file into Graphitti: Done by the `GraphManager` class described below.

### GraphManager

The GraphManager is mainly a wrapper around the Boost Graph Library (BGL) but you should not need to know about Boost to use it. The BGL loads the properties for the graph, vertices and edges into user defined structs. We have declared the `VertexProperty`, `EdgeProperty`, and `GraphProperty` structs for that purpose in the `Global.h` file. The GraphManager needs to convert each property into the right type and load them into the appropriate struct member variable. We tell GraphManager where to load the properties via the `registerProperty()` method and it infers the appropriate type. The registration of the Graph properties is being implemented as an OperationManager step that is called in the Driver class before reading the GraphML file, therefore classes that need to load graph properties are responsible for implementing the `registerGraphProperties()` method. The following is the `Layout911` implementation:

```cpp
void Layout911::registerGraphProperties()
{
   // The base class registers properties that are common to all vertices
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
```

Note: The second argument of the `registerProperty()` methods is a [pointer to a data member](https://www.studytonight.com/cpp/pointer-to-members.php) which the BGL uses for inferring the property data types and assign them to the right object's variable.


## Event Input File

