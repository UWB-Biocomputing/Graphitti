## Initial Simulator System Graph

Graphitti being a graph-based simulator works with systems that can be expressed as a network of vertices (nodes) and edges (connections). The first models developed with Graphitti were Biological Neural Network ones, for many of these models the network starts as a set of disconnected neurons laid out in a grid structure that defines their X and Y locations. These neurons then form connections as the simulation goes on. Feeding the set of disconnected neurons to Graphitti was relatively simple since all we needed was a list of identifiers. Neuron attributes, such as neuron type, were provided through separate files that contained a list of neuron identifiers for each attribute which worked well for neural network models. Unfortunately, it doesn't scale well for networks such as the NG911 with a predefined complex structure.

The edges between nodes in an NG911 network depend on their geographical location and service boundaries (or jurisdictional boundaries). We have decided that the best way to feed such a network to Graphitti is through a configuration file that contains the list of vertices and edges along with their attributes. We also chose to use a standardized file format, settling for [GraphML](http://graphml.graphdrawing.org/) because it is supported by the [Boost Graph Library](https://www.boost.org/doc/libs/1_81_0/libs/graph/doc/index.html), and other libraries and projects.

There are two main steps to implementing a graph representation of an NG911 network for Graphitti:

1. [Extracting the Graph from the entities' geographical location and service boundaries](../Tools/GIStoGraph.md), and
2. Loading the Network from a GraphML file into Graphitti: Done by the `GraphManager` class described below.

### GraphManager Class

The GraphManager is mainly a wrapper around the Boost Graph Library (BGL) but you should not need to know about Boost to use it. The BGL loads the properties for the graph, vertices and edges into user-defined structs. We have declared the `VertexProperty`, `EdgeProperty`, and `GraphProperty` structs for that purpose in the `Global.h` file. The GraphManager needs to convert each property into the right type and load them into the appropriate struct member variable. We tell GraphManager where to load the properties via the `registerProperty()` method and it infers the appropriate type. The registration of the Graph properties is being implemented as an OperationManager step that is called in the Driver class before reading the GraphML file, therefore classes that need to load graph properties are responsible for implementing the `registerGraphProperties()` method. The following is the `Layout911` implementation:

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

Note: The second argument of the `registerProperty()` method is a [pointer to a data member](https://www.studytonight.com/cpp/pointer-to-members.php) which the BGL uses for inferring the property data types and assigning them to the right object's variable.


## Event Inputs

Graphitti models the system behavior as a discrete sequence of events. In the case of the Biological Neural Network model, these are spikes that occur once a neuron's membrane electrical charge reaches a threshold (spiking neuron model). In the NG911 model, events are calls or other information being transmitted between entities. The logic of the system behavior is coded into the Simulator classes, but there are instances in which we want to be able to feed events as external inputs. For example, recreate a past event from a 911 call log, or to simulate external neuro stimuli.

The list of events is provided to Graphitti as an XML input file. We couldn't find a standardized format so we have defined our own, which consists of an XML file with a root node `<simulator_inputs>` and a `<data>` node that contains a list of events organized by `vertex`. Other nodes can be placed at the `<data>` level for general input properties if required by a model. The following is an example of an NG911 input file:


```xml
<?xml version='1.0' encoding='UTF-8'?>
<simulator_inputs>
  <data description="SPD_calls_sept2020" clock_tick_size="1sec">
    <vertex id="194" name="SEATTLE PD Caller region">
      <event time="0" duration="0" x="-122.38496236371942" y="47.570236838209546" type="EMS"/>
      <event time="34" duration="230" x="-122.37482094435583" y="47.64839548276973" type="EMS"/>
      <event time="37" duration="169" x="-122.4036487601129" y="47.55833788618255" type="Fire"/>
      <event time="42" duration="327" x="-122.38534886929502" y="47.515324716436346" type="Fire"/>
...
    <vertex/>
...
  <data/>
<simulator_inputs/>
```

An input file for the Biological Neural Network model only needs the `time` and `vertex id` such as in the following example:

```xml
<?xml version='1.0' encoding='UTF-8'?>
<simulator_inputs>
  <data description="neuro_input_test" clock_tick_size="1usec">
    <vertex id="1">
      <event time="0" vertex_id="1"/>
      <event time="34" vertex_id="1"/>
      <event time="47" vertex_id="1"/>
      <event time="73" vertex_id="1"/>
    </vertex>
    <vertex id="2">
      <event time="130" vertex_id="2"/>
      <event time="324" vertex_id="2"/>
      <event time="388" vertex_id="2"/>
      <event time="401" vertex_id="2"/>
    </vertex>
  </data>
</simulator_inputs>
```

It is important to note that some attributes such as the data description and vertex name have been added for human readability and are not used by the Simulator.

The reading of the inputs into the simulator is done similarly to reading the graph network from a GraphML file via the InputManager class.

### InputManager Class

This class was designed to work similarly to the `GraphManager` class. Event properties have to be registered to a Struct member variable that the InputManager then uses for assigning to the correct object attribute and to infer the data type. To do so, the class makes use of `boost::variant` and a properties `map`.

Unlike GraphManager, the InputManager is a template class that works with user-defined Structs. Two Structs with an inheritance relationship have been defined in the `Event.h` file: Event (for neural networks) and Call (for NG911) that inherits from Event. But we can use any Struct if we register the properties with it.

Only the data types in the `boost::variant` template parameters will work; if you have to use a Struct member with a data type different from `int, uint64_t, long, float, double, or string` then you will need to add the data type as a template parameter to the `boost::variant` and add the logic to convert the XML strings into the given type.

Examples of how to use the InputManager class can be found in the `InputmanagerTests.cpp` unit tests. The following is a simple snippet taken from a unit test that reads the events in the previous example of a neural network input file:

```cpp
#include "InputManager.h"
#include "Event.h"

string neuroInputs = "../Testing/TestData/neuro_inputs.xml"

TEST(InputManager, readNeuroInputs) {
    InputManager<Event> inputManager;
    inputManager.setInputFilePath(neuroInputs);

    // Register event properties
    inputManager.registerProperty("vertex_id", &Event::vertexId);
    inputManager.registerProperty("time", &Event::time);

    inputManager.readInputs();

    ASSERT_FALSE(inputManager.queueEmpty(1));
    auto eventList = inputManager.getEvents(1, 0, 74);
    ASSERT_EQ(eventList.size(), 4);

    // Check all events in the list
    ASSERT_EQ(eventList[0].vertexId, 1);
    ASSERT_EQ(eventList[0].time, 0);

    ASSERT_EQ(eventList[1].vertexId, 1);
    ASSERT_EQ(eventList[1].time, 34);

    ASSERT_EQ(eventList[2].vertexId, 1);
    ASSERT_EQ(eventList[2].time, 47);

    ASSERT_EQ(eventList[3].vertexId, 1);
    ASSERT_EQ(eventList[3].time, 73);
}
```

**Note**: Here we set the input file path directly for testing purposes but the `InputManager` class will get the path from the ParameterManager if one has not been set.