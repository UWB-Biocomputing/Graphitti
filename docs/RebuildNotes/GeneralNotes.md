## New Architecture notes

#### Smart pointers 
Wherever possible, Braingrid is written using smart pointers. There are three  

#### testing
google tests use cmake 
clion compatible with cmake

#### Chain of Responsibility
chain node passes in function you want to invoke. 

#### todo: 
write story for every class. 
initialize: alloc mem, init variab, setup params, copy to/from gpu

linked list for each operation

#### Simulator class
- static object / singleton class - only one of it. getInstance() helps access object.
-


vertices::AllocateMemory


synapses update growth model. not alg that operates at level of synapse. operates at level of long term neuron activity. 

STDP synapse. (emily worked on these) ;-'

under connections: conn growth. conn static. 

two different ways of managing connections: one that modified synapse, other that 

synapse looks at times of spikes. 
stdpSynapses vs alldynamicstdpsynapses. 

alldynamicstdpsynapses right now is just for future use. 

At end we dump out weight matrix. 
ToDo: we need to do this for intermediate timesteps. weight matrix per timestep in synaptic weight recorder class. # of synapses per neuron 


#### stories for every class: 
model - decides on gpu or cpu 

connections class connects, destroys 



#### Current Student Projects: 

## For Tori Salvatore's Project:
synaptic weight recorder class
use recorder class heirarchy to attach to neurons
spike timing plasticity
patterns of connectivity
conceptual behavior of burst 

## Future Projects for Rebuild: 
Timer Class elegant solution

*******************************************************
Utils/RNG/mersenneRNGnotes

for prgm to find this, needs path. alternative is to put into subdirectory.

path for mersennetwister/data/mersennetwister.dat is done

when you install, youre going to get an executable and a directory.

RNG/MersenneTwister/MersenneTwister.dat

*************************************
cpuspikingmodel.cpp

/// Advance everything in the model one time step.
void CPUSpikingModel::advance()
{
   // ToDo: look at pointer v no pointer in params
   // ToDo: look at pointer v no pointer in params - to change
   // dereferencing the ptr, lose late binding -- look into changing!
    neurons_->advanceNeurons(*synapses_, synapseIndexMap_);
    synapses_->advanceSynapses(neurons_, synapseIndexMap_);
}

*******************************************************


Lizzy param notes in core

TinyXPath is replacing TinyXML
both directories are going to be in TinyXPath

Reviewing parameterManager.h

normally the path needs root down but // gets mid root

make parammgr singleton

a client needs a single line of code. key value pair. key for param. var by ref. gets back result.

simulators generally have lots of things that need to be accessed in a lot of places.

sim object singleton. access via class name. special case of global variable.

lots of info like that. param mgr is example.

have a bunch of singletons?

all read param methods go away. keep print ops? put in chain of responsibility?

todo: check that each neuron id is actually valid

#38 add params and required dependencies.

*******************************************************


7/29 - stiver meeting on Lizzy’s notes 
Each concrete class needs to implement a static create method because it needs to be created before obj exists. return smart ptr. Unique or shared? Shared ptr probably. Multiple places need to access objects. 

New or smart ptr create class? Decide. 

Lizzy has constructor for factory register all neurons. 

Could call factory class and assign function. 

Should be able to write results file at end of simulation. - it would be good to check at the beginning to make sure you can write to a file. There are standard ways to open directory at beginning and check if you have writing permissions. 


Driver stiff: 
stateinfile Only needed by driver not sim. Param file 

stateoutfile to delete. 
stiminfile used by input classes. - needed by simulator. 
Meminfile - deserialization file- needed by simulator. 
Memoutfile - serialization file. - needed by simulator. 

Keep parse command line. 
Serialize/decerealize goes away
Terminate goes away 

Start implementing factories and 

Make proper connections classes. 

To get 
Factory working, classes working

Then able to run a simulation. 




