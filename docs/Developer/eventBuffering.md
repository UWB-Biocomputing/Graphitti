## Spike Recording by the Neuro Vertex Classes

Some of the vertex classes in the neuro domain produce *spikes*: short-lived pulses in the biological system that are modeled here as events that happen at points in time. (Note that it may be the case that other domains produce events and so we may want to generalize this to "event recording" eventually.) Spike recording is necessary for basic simulator functions, such as transmittal of spikes to postsynaptic neurons, calculation of neurite outgrowth/retraction, or computation of spike timing dependent plasticity (STDP).

### `AllSpikingNeurons`
The `AllSpikingNeurons` class is a subclass of the top-level `AllVertices` class. Relevant data members are (**note: these are all public; any changes should fix this**):

- `hasFired_`: dynamically allocated `bool` array indicating that a neuron has produced a spike during this time step.
- `spikeCount_`: dynamically allocated `int` array; number of spikes neuron has produced during this epoch.
- `spikeCountOffset_`: dynamically allocated `int` array
- `spikeHistory_`: dynamically allocated; type is `uint64_t**`; this seems to be an array of buffers holding the time step for each spike (one buffer per neuron)

Each of these variables are also present in the `AllSpikingNeuronsDeviceProperties` struct, an extension of the `AllVerticesDeviceProperties` struct, which gets allocated on the GPU.

#### Algorithms relevant to data members

##### `hasFired_`
- Nulled out by `AllSpikingNeurons::AllSpikingNeurons()`
- deleted and nulled out by `AllSpikingNeurons::~AllSpikingNeurons()`
- Storage allocated (dimension: `size_`) and initialized to false by `AllSpikingNeurons::setupVertices()`
- Primarily manipulated by `AllSpikingNeurons::advanceVertices()`:
  * Tests if true and, if so:
    + Calls `AllSpikingSynapses::preSpikeHit()` for each outgoing synapse index.
    + (**Note: not clear if the following is actual usable code.**) If `AllSpikingSynapses::allowBackPropagation()` returns true, then calls `AllSpikingSynapses::postSpikeHit(() for each incoming synapse index.
    + Sets `hasFired_` to false
  * Set to true by `AllSpikingNeurons::fire()`, which is called by a subclass implementation of `advanceNeuron()` (via a subclass implementation of `fire()`)

##### `spikeCount_`
This is the number of spikes produced by a neuron during an epoch. It is also the offset within the `spikeHistory_` of a neuron corresponding to one past the end of the circular buffer (i.e., one past the end of the queue; where the next spike will be enqueued).
- Nulled out by `AllSpikingNeurons::AllSpikingNeurons()`
- deleted and nulled out by `AllSpikingNeurons::~AllSpikingNeurons()`
- Storage allocated (dimension: `size_`) and initialized to 0 by `AllSpikingNeurons::setupVertices()`
- Set to 0 by `AllSpikingNeurons::clearSpikeCounts()`
  * This method is called by `XmlRecorder::compileHistories()` and `Hdf5Recorder::compileHistories()`
- `AllSpikingNeurons::advanceVertices()` tests that it is less than the maximum number of spikes per epoch
- Used by `AllSpikingNeurons::fire()`:
  * to compute the index of a spike in `spikeHistory_` to write the simulation time step
  * after that, it is incremented, so this is the method that counts the spikes
- Used by `AllSpikingNeurons::getSpikeHistory()` to retrieve the simulation time step for a particular spike and neuron. This is an accessor for the spike history data structures, but is used for accessing preceding spikes, i.e., the most recent spike, two spikes ago, three spikes ago, etc. It would be awkward to use this to grab an epoch's worth of spikes.
  * This method is called by the 911 and STDP edge/synapse classes `advanceEdge()` methods to retrieve spikes from preceding time steps.

##### `spikeCountOffset_`
This is the start of the circular buffer (front of the queue) in `spikeHistory_` associated with a neuron.
- Nulled out by `AllSpikingNeurons::AllSpikingNeurons()`
- deleted and nulled out by `AllSpikingNeurons::~AllSpikingNeurons()`
- Storage allocated (dimension: `size_`) and initialized to 0 by `AllSpikingNeurons::setupVertices()`
- Increased by `spikeCount_` for each neuron, mod the maximum number of spikes per epoch, in `AllSpikingNeurons::clearSpikeCounts()`. See above for calling information. This resets the beginning index for the epoch within the circular buffer.
- Used by `AllSpikingNeurons::fire()` to compute the index of a spike in `spikeHistory_` to write the simulation time step
  * This index is computed by adding the value of `spikeCountOffset_` for a neuron to `spikeCount_`, mod the maximum number of spikes per epoch. This is consistent with the idea that each array in `spikeHistory_` implements a circular buffer. **This is a circular buffer because some edge algorithms need to access preceding spikes (i.e., previously-produced spikes). This might require accessing spikes produced in the preceding epoch.**
- Used by `AllSpikingNeurons::getSpikeHistory()` to retrieve the simulation time step for a particular spike and neuron.
  * This computes the index of a spike in the past (i.e., the most recent spike, two spikes ago, etc). It starts with `spikeCountOffset_ + spikeCount_`, which I believe at the end of an epoch should be one past the end of that neuron's spikes in `spikeHistory_`. Then, the maximum number of spikes per epoch is added. Then, the `offIndex` parameter is added. The expectation is that `offIndex` will be a negative number (i.e., a spike in the past); the reason that the max spikes value is added is to prevent this from producing a negative total, so that finally taking mod max spikes will "wrap around backwards" if needed.

##### `spikeHistory_`
These are the circular spike buffers for each neuron. It is an array of pointers to arrays of `uint_64t`.
- Nulled out by `AllSpikingNeurons::AllSpikingNeurons()`
- deleted (each circular buffer, then the main array) and nulled out by `AllSpikingNeurons::~AllSpikingNeurons()`
- Storage allocated (first dimension: `size_`; second dimension: zero, arrays not allocated) and elements initialized to `nullptr` by `AllSpikingNeurons::setupVertices()`
- Note that the secondary, circular buffers are allocated by `AllIFNeurons::createNeuron()`. **It's not clear why this is allocated by the subclass.**
  * Buffer size is maximum number of spikes per epoch
  * Buffer contents initialized to `ULONG_MAX`


### Refactor/Redesign Ideas

It seems like we need to retain a circular buffer for a neuron's spikes, so that it will be possible to "look backwards" across an epoch boundary to retrieve spikes from the preceding epoch(s). We should refactor this out as a separate class, `EventBuffer`:
- Two methods for use by the Recorder classes:
  * `EventBuffer::operator[]`: retrieve an event time step at an offset relative to the start of the current epoch (i.e., `0..numEvents_-1`).
  * `EventBuffer::getNumEventsInEpoch()`: return the number of events in the current epoch.
- Methods for use by the Vertex and Edge classes:
  * `EventBuffer::clear()`: resets everything (may not be useful after start of simulation)
  * `EventBuffer::startNewEpoch()`: resets variables that track where the current buffer starts and how many spikes there are in the current epoch
  * `EventBuffer::insertEvent(uint64_t timeStep)`: record (enqueue) an event at the indicated time step
  * `EventBuffer::getPastEvent(offset: int)`: return event `offset` in the past, where `offset == -1` means the most recent event, `offset == -2` is two events ago, etc.
  * Methods to copy an `EventBuffer` to and from the GPU (need to work through this; at least, it seems like this mechanism will mean that there is no need to modify the GPU code).

Then, the set of spike buffers can be allocated as a `vector` of `EventBuffer` (note that the `vector` constructor with size and initialization and `vector::resize()` with size and initialization should still work with the `EventBuffer` constructor). In the recorder classes, we can just iterate through the list (`vector`) of probed vertices to get the vertex IDs (indices) of the `EventBuffer`s to record. This will then be one of the standard variables registrations with the recorder classes: a call to register the vertex event buffers, which expects a `const vector<EventBuffer>&`. The recorders create a `vector` from the probed vertex list (when loaded by `loadParameters()`). Each epoch, the use the probed vertex `vector` to walk through the `vector` of `EventBuffer`s, in turn using `EventBuffer::operator[]` to pull out the spike time steps.

This valarray should be a data member of `AllSpikingNeurons` and its constructor should create and initialize everything. This would move the allocation and initialization out of `AllIFNeurons`.

---------
[<< Go back to the Graphitti home page](..)
