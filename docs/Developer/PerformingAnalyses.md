Refer to Hsu-MS-report20.pdf (Emily Hsu's paper) for some details
Iterating through nodes with random number generators
Assuming we haven't changed code, we will get the same sequence of random numbers. 
cpu to cpu should compare numerically. 
if there is any change numerically at least significant bits, could cause issues
81 bit floating point 
64 bit floating point
if we were comparing numbers, want to make sure rounding exactness. 

For gpu, same should be true - mersenne twister RNG
if there are n neurons, there are n generator states made. 
when we generate random numbers, each number operates on different state. 
should be same random number sequences on mersenne twister. 
only issue would be size. need to write code to do comparison for larger analyses. 

for size: 
use a diff to give side by side comparison 
could be easier to use matlab. plot vals - are any diffs > 1.0E-6

for duration: 
in configfiles : test-small.xml 10x10 grid. 100 second epoch. 2 epochs. two updates. 
initial radius is 0.4. if they are 0.2 away, they do not touch. 2 epochs don't give enough time for overlaps of neurons. 

20 epochs are enough where synapses should be created.

Are radii increasing? once they overlap, exercise synapse creation method. While andogenously active neurons produce spikes 

Being able to test something: make # of epochs long enough (10 epochs should do) and start radius large enough. 

nonlinear sigmoidal = derivative of rate of growth, where middle is zero. 

What value in output file would show that synapses are being created? 

from matlab: read matrix / radii history. 

could be possible to copy / paste 

2x2 epoch simulation. nodes are 1 unit away, thus if 0.5  is radius, it'll touch next node. 

rate history. Firing rate during preceeding epoch. used to compute change in radius. 

burstiness history - ignore for now, was created at some point and no longer used. 

spike history. = number of spikes in 10ms bin. 10ms is 100th of a second. 

100 seconds * 100 bins per second * 2 epochs. = 20,000
2014 journal paper explains this. 

starter neurons = index of andogenously active neurons. 

Neuron thresholds - what causes producing spikes 

simulation time = epoch. 

simulation end time: # epochs * epoch duration

todo: need to get readmatrix debugged / working 
HDF5 is binary format. 

cant build sim with hdf5 at the moment. 

ambition to serialize and deserialize entire state of simulation. 

common to have a shared supercomputer 
long running tasks can get booted off if there is a queue of supercomputers. need to have saved checkpoints to reload and continue computation. 

if we can have all of our data wrapped by objects, then cereal will work. 

"copy everything back from GPU" 

model of gpu could be problematic on getting exactly same results. 

what we are concerned with: if two simulations produce "similar" results 


in cortical culture analysis repository, matlab files generate plots. 


