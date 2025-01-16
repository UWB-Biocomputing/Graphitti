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
format and the class that loads and manages the event inputs](https://uwb-biocomputing.github.io/Graphitti/Developer/GraphAndEventInputs.html).

Below is an example of a simulation input file:


```XML
<?xml version='1.0' encoding='UTF-8'?>
<simulator_inputs>
  <data description="SYNTH_OUTPUT2 Calls - Cluster Point Process" clock_tick_size="1" clock_tick_unit="sec">
    <vertex id="4" name="UNKNOWN">
      <event time="26" duration="38" x="13.111984735748287" y="62.57456126541278" type="Law" patience="54" on_site_time="33" vertex_id="4"/>
      <event time="35" duration="30" x="13.101924628231098" y="62.5788162451589" type="Fire" patience="2" on_site_time="12" vertex_id="4"/>
      <event time="60" duration="48" x="13.103238844738144" y="62.57852864894537" type="Fire" patience="26" on_site_time="71" vertex_id="4"/>
      <event time="61" duration="33" x="13.103729766137114" y="62.57884701874643" type="Fire" patience="27" on_site_time="2" vertex_id="4"/>
      <event time="74" duration="32" x="13.104050198095443" y="62.578851926237746" type="Fire" patience="23" on_site_time="38" vertex_id="4"/>
      <event time="77" duration="41" x="13.125015368211848" y="62.547956222166015" type="EMS" patience="2" on_site_time="239" vertex_id="4"/>
      <event time="111" duration="45" x="13.12497529802989" y="62.54820796343649" type="EMS" patience="46" on_site_time="77" vertex_id="4"/>
      ...
    </vertex>
  </data>
</simulator_inputs>
```