# 1.4  Configuring the model

Now that you have run through a quick test and made sure you have a working Graphitti repository, it is time to learn how to use it!

We will be going through this in a few steps:

1. First, we will look at how to implement a quick and dirty model and simulation parameters, which will involve putting together all the files that Graphitti uses as inputs.

2. Second, we will configure Graphitti to use a GPU (you've already seen how to do it with a single thread). And then we will run the simulation.

3. Lastly, we will collect Graphitti's output and examine a few ways one might actually visualize the data.

Ready? Okay.

## 1.4.1 Inside the Config files

There are two config files needed to run a simulation on Graphitti:

1. The input (or "stimulation") and model configuration file - **test.xml**
2. The initialization parameters - **test.graphml**

First, we are going to go through using a built-in model. This is by far the easiest route - if you have a quick idea you want to play with that uses a setup of **Izhikivich** or **LIF** neurons, go for it! As long as you only want to use excitatory and inhibitory neurons, this is the way to go. I'll show you how to specify the parameters you want and then run the simulation.

If on the other hand, you have a more complicated model in mind - such as using different types of neurotransmitters, then you will have to get your hands dirty writing some C++ code. Don't worry though, I'll walk you through that too.

## 1.4.2 Use built-in models

Let's go through the steps required to use a built-in model.

Take a look at **test-tiny.xml** file that is under  `Graphitti/configfiles`  directory: 

```xml
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Parameter file for the DCT growth modeling -->
<!-- This file holds constants, not state information -->
<SimInfoParams>
   <!-- GraphML file that contains neuron location (x, y), type (excitatory or inhibitory),
       and activness (endogenously active or not) -->
   <graphmlFile name="graphmlFile">../configfiles/graphs/test-tiny.graphml</graphmlFile>
   <!-- Simulation Parameters -->
   <SimParams epochDuration="1.0" numEpochs="1"/>
   <!-- Simulation Configuration Parameters -->
   <SimConfig maxFiringRate="200" maxEdgesPerVertex="200"/>
   <!-- Random seed - set to zero to use /dev/random -->
  <RNGConfig name="RNGConfig">
      <InitRNGSeed name="InitRNGSeed">1</InitRNGSeed>
      <NoiseRNGSeed class="Norm" name="NoiseRNGSeed">1</NoiseRNGSeed>
   </RNGConfig>
</SimInfoParams>

<ModelParams>
   <VerticesParams class="AllLIFNeurons">
      <!-- Interval of constant injected current -->
      <Iinject min="13.5e-09" max="13.5e-09"/>
      <!-- Interval of STD of (gaussian) noise current -->
      <Inoise min="1.0e-09" max="1.5e-09"/>
      <!-- Interval of firing threshold -->
      <Vthresh min="15.0e-03" max="15.0e-03"/>
      <!-- Interval of asymptotic voltage -->
      <Vresting min="0.0" max="0.0"/>
      <!-- Interval of reset voltage -->
      <Vreset min="13.5e-03" max="13.5e-03"/>
      <!-- Interval of initial membrance voltage -->
      <Vinit min="13.0e-03" max="13.0e-03"/>
      <!-- Starter firing threshold -->
      <starter_vthresh min="13.565e-3" max="13.655e-3"/>
      <!-- Starter reset voltage -->
      <starter_vreset min="13.0e-3" max="13.0e-3"/>
   </VerticesParams>
   
   <EdgesParams class="AllDSSynapses" name="EdgesParams">
   </EdgesParams>
   
   <ConnectionsParams class="ConnGrowth">
      <!-- Growth parameters -->
      <GrowthParams epsilon="0.60" beta="0.10" rho="0.0001" targetRate="1.9" minRadius="0.1" startRadius="0.4"/>
   </ConnectionsParams>

   <!-- This may be empty, but it is needed to create the classes/objects for the simulator -->
   <LayoutParams class="LayoutNeuro" name="LayoutParams">
            <LayoutFiles name="LayoutFiles">
            </LayoutFiles>
   </LayoutParams>

</ModelParams>
```

This is a typical example of a model configuration file that you must give to use Graphitti. This type of file is mandatory - Graphitti won't run without specifying model parameters. Even if you plan on writing your own model from "scratch", you may want to read this section anyway.

You can see that this file is a pretty standard XML file. It has tags that specify what each section is, like `<SimInfoParams>` and end tags that end said section, like `</SimInfoParams>`. Within each section, you can have sub-sections ad infinitum (in fact, XML files follow a tree structure, with a root node which branches into a top level of nodes, which branch into their own nodes, which branch, etc.)

Notice the `<!-- Parameter file for the DCT growth modeling -->` Anything that follows that pattern (i.e., `<!-- blah blah blah -->`) is a comment, and won't have any effect on anything. It is good practice to comment stuff in helpful, far-seeing ways.

On to the actual parameters.

#### SimInfoParams

The first set of parameters that Graphitti expects out of this file is stored in the SimInfoParams node. These parameters are required no matter what your model is. Here you must specify the:

* **graphmlFile**: GraphML file that holds data for the initial structure of the simulation, such as the number of neurons, their xy locations, type, and activeness. We will go into further detail later on.
* **SimParams**: the time configurations - expects a epochDuration, which is how much time the simulation is simulating (in seconds) and a numEpochs, which is how many times to run the simulation (each simulation cycle picks up where the previous one left off)
* **SimConfig**: the maxFiringRate of a neuron and the maxEdgesPerVertex (the limitations of the simulation). Note the rate is in Hz.
* **Seed**: a random seed for the random generator.
* **OutputParams**: requires stateOutputFileName, which is where the simulator will store the output file.

#### ModelParams

The next set of parameters is the ModelParams. These parameters are specific to your model. Later, when we go through the "from scratch" example (where you will code up your own model using C++ to provide utmost flexibility), you will specify what goes here. But for now, we are using a built in model - specifically LIF (leaky integrate and fire), just to see what's expected. You must specify the:

* **VerticesParams**: This is an XML node in and of itself, which requires several items. Each of these items is presented as a range, with the idea that each neuron will be chosen with random values from each of these intervals.
    + **Iinject**: The interval of constant injected current. Each neuron will be randomly assigned a value from this interval on start (with a uniform distribution).
    + **Inoise**: Describes the background (noise) current, if you want some in your experiment (simulates realistic settings). Each neuron will have a background noise current chosen from this range.
    + **Vthresh**: The threshold membrane voltage that must be reached before a neuron fires; again, specified as a range of values from which each neuron will be chosen randomly.
    + **Vresting**: The resting membrane potential of a neuron.
    + **Vreset**: The voltage that a neuron gets reset to after firing.
    + **Vinit**: The starting voltage of a neuron.
    + **starter_vthresh**: In this particular model, there are endogenously active neurons called 'starter neurons', whose threshold voltage is drawn from this range. This range is set low enough that their noise can actually drive them to fire, so that there need not be any input into the neural net. You can of course, configure these neurons to be exactly the same as the other ones, but without then coding an input to the net, your net won't do anything.
    + **starter_vreset**: The voltage to which a starter neuron gets reset after firing.

* **EdgesParams**: Another node that should be populated - though you'll note in this particular example, we aren't specifying anything about the synapses.

* **ConnectionsParams**: Another node to populate. Its parameters are as follows:
    + **GrowthParams**: The growth parameters for this simulation. The mathematics behind epsilon, beta, and rho can be found [TODO]. The targetRate is TODO, and the minRadius, and startRadius should be self-explanatory.

* **Layout Params**: Currently empty as the GraphML file holds this information. This is still kept in the file so that the simulator can create the class.

### GraphML
You might have noticed that within **SimInfoParams** there is a parameter that points to another file. This is the GraphML file, a type of XML file that is used to describe graphs through node, edge, and graph descriptors. For more information about how GraphML works, [see this page](http://graphml.graphdrawing.org/primer/graphml-primer.html). In the context of the Neural simulation, the GraphML file holds details about neuron contents. Lets look at the GraphML file mentioned in the xml, which is also within the `Graphitti/configfiles/graphs` directory:

```xml
<?xml version='1.0' encoding='utf-8'?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
  <key id="type" for="node" attr.name="type" attr.type="string" />
  <key id="active" for="node" attr.name="active" attr.type="long" />
  <key id="y" for="node" attr.name="y" attr.type="long" />
  <key id="x" for="node" attr.name="x" attr.type="long" />
  <graph edgedefault="directed">
    <node id="0">
      <data key="x">0</data>
      <data key="y">0</data>
      <data key="active">1</data>
      <data key="type">EXC</data>
    </node>
    <node id="1">
      <data key="x">1</data>
      <data key="y">0</data>
      <data key="active">0</data>
      <data key="type">INH</data>
    </node>
    <node id="2">
      <data key="x">0</data>
      <data key="y">1</data>
      <data key="active">0</data>
      <data key="type">EXC</data>
    </node>
    <node id="3">
      <data key="x">1</data>
      <data key="y">1</data>
      <data key="active">0</data>
      <data key="type">EXC</data>
    </node>
  </graph>
</graphml>
```

As you can see, there are four neurons, or nodes, and each neuron has four keys that store its x,y location, neuron type, and whether it is endogenously active or not. There are no edges because the GraphML file is used in initialization, and the initial state of the simulation has no connections between the neurons.

-------------
[<< Go back to User Documentation page](index.md)

---------
[<< Go back to Graphitti home page](http://uwb-biocomputing.github.io/Graphitti/)
