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
  <key id="active" for="node" attr.name="active" attr.type="boolean" />
  <key id="y" for="node" attr.name="y" attr.type="double" />
  <key id="x" for="node" attr.name="x" attr.type="double" />
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

## 1.4.3 911

Recently we added an additional domian to our simulator, 911 communication networks. The configuration for these simulations are the same as the neural network simulations in the sense that they both have an xml configuration file and a GraphML file (the parameters are different, obviously). We discussed the neural simulation parameters above, so in this section we will go over the 911 specific details. We start off with the xml configuration file, then the GraphML file. The 911 simulation has a third configuration file, which are the inputs ("calls"), so we go over that last.

### Main Configuration

There are some elements of the main configuration file that is similar to the Nerual simulation, such as **SimParams**, **SimConfig**, and **RNGConfig**. The only vertice parameters for the 911 simulation are RedialPropability, which determines how likely a call will happen again, and the average driving speed of a response unit. An example configuration file is shown below.

```xml
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<BGSimParams>
  <SimInfoParams name="SimInfoParams">
    <SimParams name="SimParams">
      <!-- 2 epochs of 15 minutes each -->
      <epochDuration name="epochDuration">900</epochDuration>
      <numEpochs name="numEpochs">2</numEpochs>
      <!-- Every simulation step is 1 second -->
      <deltaT name="Simulation Step Duration">1</deltaT>
    </SimParams>
    <SimConfig name="SimConfig">
      <maxFiringRate name="maxFiringRate">100</maxFiringRate>
      <!-- max edges in a vertex for test-small-911.graphml is 20 -->
      <maxEdgesPerVertex name="maxEdgesPerVertex">25</maxEdgesPerVertex>
    </SimConfig>
    <RNGConfig name="RNGConfig">
      <InitRNGSeed name="InitRNGSeed">1</InitRNGSeed>
      <NoiseRNGSeed class="Norm" name="NoiseRNGSeed">1</NoiseRNGSeed>
    </RNGConfig>
  </SimInfoParams>

  <ModelParams>
    <VerticesParams class="All911Vertices" name="VerticesParams">
      <RedialP name="RedialProbability">0.85</RedialP>
      <!-- Response unit driving speed in mph -->
      <AvgDrivingSpeed name="AverageDrivingSpeed">30.0</AvgDrivingSpeed>
    </VerticesParams>

    <EdgesParams class="All911Edges" name="EdgesParams">
    </EdgesParams>

    <ConnectionsParams class="Connections911" name="ConnectionsParams">
      <graphmlFile name="graphmlFile">../configfiles/graphs/test-small-911.graphml</graphmlFile>
      <psapsToErase name="psapsToErase">0</psapsToErase>
      <respsToErase name="respsToErase">0</respsToErase>
    </ConnectionsParams>

    <InputParams name="InputParams">
      <inputFile name="inputFile">../configfiles/inputs/test-small-911-calls.xml</inputFile>
    </InputParams>

    <LayoutParams class="Layout911" name="LayoutParams">
    </LayoutParams>
    
    <RecorderParams class="Xml911Recorder" name="RecorderParams">
      <RecorderFiles name="RecorderFiles">
        <resultFileName name="resultFileName">Output/Results/test-small-911-out.xml</resultFileName>
      </RecorderFiles>
    </RecorderParams>
  </ModelParams>
</BGSimParams>
```

### GraphML

The 911 GraphML file holds more data than the neural GraphML file. Both hold (x, y) location and node type, but the 911 graph also holds data such as ID, node name, number of servers, etc. The 911 GraphML also has edges, since these are defined from the start. The corresponding GraphML file for the configuration file above is:

```xml
<?xml version='1.0' encoding='utf-8'?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
<key id="segments" for="node" attr.name="segments" attr.type="string"/>
<key id="trunks" for="node" attr.name="trunks" attr.type="int"/>
<key id="servers" for="node" attr.name="servers" attr.type="int"/>
<key id="x" for="node" attr.name="x" attr.type="double"/>
<key id="y" for="node" attr.name="y" attr.type="double"/>
<key id="type" for="node" attr.name="type" attr.type="string"/>
<key id="name" for="node" attr.name="name" attr.type="string"/>
<key id="objectID" for="node" attr.name="objectID" attr.type="string"/>
<graph edgedefault="directed">
<node id="0">
  <data key="objectID">PSAP120@kingcounty.gov</data>
  <data key="name">SEATTLE PD</data>
  <data key="type">PSAP</data>
  <data key="y">47.604309311762556</data>
  <data key="x">-122.32903357244062</data>
  <data key="servers">4</data>
  <data key="trunks">5</data>
</node>
<node id="1">
  <data key="objectID">EMS218@kingcounty.gov</data>
  <data key="name">Seattle FD - Greenwood</data>
  <data key="type">EMS</data>
  <data key="y">47.68208718681924</data>
  <data key="x">-122.35498346016459</data>
  <data key="servers">5</data>
  <data key="trunks">10</data>
</node>
<node id="2">
  <data key="objectID">EMS246@kingcounty.gov</data>
  <data key="name">Seattle FD - West Seattle</data>
  <data key="type">EMS</data>
  <data key="y">47.560795490890584</data>
  <data key="x">-122.37975380249847</data>
  <data key="servers">3</data>
  <data key="trunks">10</data>
</node>
<node id="3">
  <data key="objectID">EMS262@kingcounty.gov</data>
  <data key="name">Seattle FD - Laurelhurst</data>
  <data key="type">EMS</data>
  <data key="y">47.66863807147154</data>
  <data key="x">-122.28446260064581</data>
  <data key="servers">3</data>
  <data key="trunks">10</data>
</node>
<node id="4">
  <data key="objectID">Law104@kingcounty.gov</data>
  <data key="name">Seattle PD - North Precinct</data>
  <data key="type">LAW</data>
  <data key="y">47.70290245181223</data>
  <data key="x">-122.33454218929678</data>
  <data key="servers">3</data>
  <data key="trunks">10</data>
</node>
<node id="5">
  <data key="objectID">Law108@kingcounty.gov</data>
  <data key="name">Seattle PD - West Precinct</data>
  <data key="type">LAW</data>
  <data key="y">47.61615143092955</data>
  <data key="x">-122.33703384492664</data>
  <data key="servers">3</data>
  <data key="trunks">10</data>
</node>
<node id="6">
  <data key="objectID">Law109@kingcounty.gov</data>
  <data key="name">Seattle PD - East Precinct</data>
  <data key="type">LAW</data>
  <data key="y">47.615172448270286</data>
  <data key="x">-122.31711730383483</data>
  <data key="servers">3</data>
  <data key="trunks">5</data>
</node>
<node id="7">
  <data key="objectID">Law119@kingcounty.gov</data>
  <data key="name">Seattle PD - Southwest Precinct</data>
  <data key="type">LAW</data>
  <data key="y">47.53601093999662</data>
  <data key="x">-122.3619473720436</data>
  <data key="servers">4</data>
  <data key="trunks">10</data>
</node>
<node id="8">
  <data key="objectID">FIRE169@kingcounty.gov</data>
  <data key="name">Seattle FD - Beacon Hill</data>
  <data key="type">FIRE</data>
  <data key="y">47.571870954923924</data>
  <data key="x">-122.30856118715617</data>
  <data key="servers">5</data>
  <data key="trunks">10</data>
</node>
<node id="9">
  <data key="objectID">FIRE218@kingcounty.gov</data>
  <data key="name">Seattle FD - Greenwood</data>
  <data key="type">FIRE</data>
  <data key="y">47.68208718681924</data>
  <data key="x">-122.35498346016459</data>
  <data key="servers">3</data>
  <data key="trunks">10</data>
</node>
<node id="10">
  <data key="objectID">PSAP120@kingcounty.gov_CR</data>
  <data key="name">SEATTLE PD Caller region</data>
  <data key="type">CALR</data>
  <data key="segments">[(-122.40326150882478, 47.59789698297564), (-122.37990573296226, 47.62125275883817)], [(-122.47332883641236, 47.55118543125059), (-122.44997306054984, 47.574541207113114)], ...</data>
</node>
<edge source="0" target="1"/>
<edge source="0" target="2"/>
<edge source="0" target="3"/>
<edge source="0" target="4"/>
<edge source="0" target="5"/>
<edge source="0" target="6"/>
<edge source="0" target="7"/>
<edge source="0" target="8"/>
<edge source="0" target="9"/>
<edge source="0" target="10"/>
<edge source="1" target="0"/>
<edge source="2" target="0"/>
<edge source="3" target="0"/>
<edge source="4" target="0"/>
<edge source="5" target="0"/>
<edge source="6" target="0"/>
<edge source="7" target="0"/>
<edge source="8" target="0"/>
<edge source="9" target="0"/>
<edge source="10" target="0"/>
</graph></graphml>
```

### Inputs

The final file used in 911 configuration is the input file, which is a simple xml file that describes when an input (call) is introduced into the simulation, the location of the input, the type of call, its duration, etc. The file is shown below.

```xml
<?xml version='1.0' encoding='UTF-8'?>
<simulator_inputs>
  <data description="SPD_calls_sept2020" clock_tick_size="1" clock_tick_unit="sec">
    <vertex id="10" name="SEATTLE PD Caller region">
      <event time="34" duration="230" x="-122.37482094435583" y="47.64839548276973" type="EMS" vertex_id="10" patience="61" on_site_time="3142"/>
      <event time="37" duration="169" x="-122.4036487601129" y="47.55833788618255" type="Fire" vertex_id="10" patience="3" on_site_time="2032"/>
      <event time="42" duration="327" x="-122.38534886929502" y="47.515324716436346" type="Fire" vertex_id="10" patience="8" on_site_time="782"/>
      <event time="47" duration="165" x="-122.27568876640863" y="47.67904232558008" type="EMS" vertex_id="10" patience="9" on_site_time="627"/>
      <event time="73" duration="262" x="-122.36701587581143" y="47.51932484119922" type="EMS" vertex_id="10" patience="50" on_site_time="1890"/>
      <event time="130" duration="242" x="-122.32733110385414" y="47.65708716342721" type="Law" vertex_id="10" patience="150" on_site_time="1105"/>
      <event time="324" duration="209" x="-122.42842939223208" y="47.59298276266974" type="Fire" vertex_id="10" patience="5" on_site_time="603"/>
      <event time="388" duration="45" x="-122.37746466732693" y="47.711139673719046" type="Law" vertex_id="10" patience="66" on_site_time="182"/>
      <event time="401" duration="110" x="-122.45031189490172" y="47.704142615892934" type="EMS" vertex_id="10" patience="45" on_site_time="53"/>
      <event time="435" duration="54" x="-122.38497853357659" y="47.58597687545961" type="EMS" vertex_id="10" patience="139" on_site_time="770"/>
      <event time="490" duration="259" x="-122.33562990965024" y="47.64880090244549" type="Law" vertex_id="10" patience="77" on_site_time="2142"/>
      <event time="541" duration="350" x="-122.37503877878919" y="47.5274803656548" type="Law" vertex_id="10" patience="86" on_site_time="510"/>
      <event time="671" duration="389" x="-122.43846764661109" y="47.594429736848575" type="EMS" vertex_id="10" patience="60" on_site_time="637"/>
      <event time="900" duration="81" x="-122.29491177376885" y="47.60886297661482" type="Fire" vertex_id="10" patience="57" on_site_time="90"/>
      <event time="960" duration="53" x="-122.29648552268789" y="47.61776320232366" type="Law" vertex_id="10" patience="113" on_site_time="824"/>
      <event time="1009" duration="638" x="-122.35596534749182" y="47.6773160725642" type="Law" vertex_id="10" patience="38" on_site_time="81"/>
      <event time="1106" duration="230" x="-122.44721574783861" y="47.60935496208115" type="EMS" vertex_id="10" patience="16" on_site_time="3378"/>
      <event time="1110" duration="64" x="-122.31713723952292" y="47.54535782341512" type="Fire" vertex_id="10" patience="8" on_site_time="139"/>
      <event time="1143" duration="241" x="-122.39378273701678" y="47.48431540296322" type="Fire" vertex_id="10" patience="47" on_site_time="633"/>
      <event time="1155" duration="180" x="-122.43233093624063" y="47.682969737450534" type="Fire" vertex_id="10" patience="20" on_site_time="967"/>  
    </vertex>
  </data>
</simulator_inputs>
```

-------------
[<< Go back to User Documentation page](index.md)

---------
[<< Go back to Graphitti home page](http://uwb-biocomputing.github.io/Graphitti/)
