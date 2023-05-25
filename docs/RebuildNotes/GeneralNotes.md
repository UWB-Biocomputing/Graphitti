## Object Notes
### summationMap
summation points for a vertex. setup() methods are allocating internal storage for summationMap

## Methood Notes

### setup()
 setup is copying global variables relevant to a specific class.
 parameters specific to a class are not being grabbed by setup; the method only requires the object to be setup.

### setupVertices()




## New Architecture notes

#### Smart pointers 
Wherever possible, Graphitti is written using smart pointers. There are three  

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

## Future Projects for Rebuild: 
Timer Class elegant solution

*******************************************************




make parammgr singleton

a client needs a single line of code. key value pair. key for param. var by ref. gets back result.

sim object singleton. access via class name. special case of global variable.


Should be able to write results file at end of simulation. - it would be good to check at the beginning to make sure you can write to a file. There are standard ways to open directory at beginning and check if you have writing permissions. 







