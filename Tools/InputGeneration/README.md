# Simulation Inputs

The behavior of the system during a simulation is dictated by the occurrence
of sequences of events. These events can represent a neuronal electric impulse
(spike) in a Biological Neural Network; or an emergency call or dispatch in
an Emergency Services Communication System (ESCS). These events can be part of
the internal workings of the system, or be external stimuli fed via simulation
inputs. Such input must be provided through an XML, in the format described on
the developer's documentation.

This directory contains scripts that generate simulation input files, either
by appropriately transforming real data into the required XML format or
entirely generating the inputs, synthetically.

The current input generation scripts are:

- `FromRealData/input_file_from_call_log.py`: Takes a real 911 call log and generates an
    XML input file appropriately formatted.
- `ClusterPointProcess\cluster_point_process.py`: Generates synthetic calls
    modeled as a spatio-temporal cluster point process.

## Cluster Point Process

The call arrival process is an important and complex part of modeling Emergency
Services Communication Systems (ESCS). Starting with the idea that emergency calls
result from incidents, we can depict call arrivals as a cluster point process. Here,
emergency calls gather around emergency incidents.

The `cluster_point_process.py` script generates emergency calls, modeled as a
spatio-temporal cluster point process. This means calls cluster in time and space around
a primary process representing emergency incidents. Currently, the primary process is a
Homogeneous Poisson Process, but it could be any reasonable stochastic or deterministic
process — stationary or non-stationary. It could even source from a stream of real incident data.

The stream of calls generated is saved to an XML file. This file contains a list of
`vertices` representing Caller Regions and their stream of `event` items (calls).

## Input file format

The developer's documentation contains more information about the [simulation input file
format and the class that loads and manages the event inputs]
(https://uwb-biocomputing.github.io/Graphitti/Developer/GraphAndEventInputs.html).

Below is an example of a simulation input file:


```XML
?xml version='1.0' encoding='UTF-8'?>
<simulator_inputs>
  <data description="SPD Calls - Cluster Point Process" clock_tick_size="1" clock_tick_unit="sec">
    <vertex id="194" name="SEATTLE PD Caller region">
      <event time="12" duration="568" x="-122.43656117361931" y="47.604800151342786" type="Law" vertex_id="194"/>
      <event time="99" duration="36" x="-122.43611438152655" y="47.5321990385178" type="EMS" vertex_id="194"/>
      <event time="336" duration="100" x="-122.34868883298704" y="47.56973909330686" type="Law" vertex_id="194"/>
      <event time="601" duration="367" x="-122.44122311134316" y="47.70298531339788" type="Law" vertex_id="194"/>
      <event time="766" duration="96" x="-122.28084792007573" y="47.70594997190379" type="Fire" vertex_id="194"/>
      ...
    </vertex>
  </data>
</simulator_inputs>
```