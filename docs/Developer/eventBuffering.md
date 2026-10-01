## Spike Recording by the Neuro Vertex Classes

Some of the vertex classes in the neuro domain produce *spikes*: short-lived pulses in the biological system that are modeled here as events that happen at points in time. (Note that it may be the case that other domains produce events and so we may want to generalize this to "event recording" eventually.) Spike recording is necessary for basic simulator functions, such as transmittal of spikes to postsynaptic neurons, calculation of neurite outgrowth/retraction, or computation of spike timing dependent plasticity (STDP).

### `EventBuffer`
Event buffer class serves as a data structure implementation that to be used by vertex classes and recorder classes. Specifically, it is a circular array based implementation of queue that holds events produced by vertex(neurons). It serves as interface for Vertex classes to allow event time steps(uint64_t value) to span epoch boundaries and vector like interface for recorder classes to provide zero-based indexing of just the events that occurred in the preceding epoch. 
Relevant data members are:
- `dataSeries_`: It is vector that holds event time steps(as uint64_t) as its elements.
- `bufferFront_`: It is an integer pointing to an index of the first event in the queue `dataSeries_`.
- `bufferEnd_`: It is an integer pointing to the location one past the end of the queue `dataSeries_` (enqueue operation is performed on pre incrementing the bufferFront_ index pointer).
- `epochStart_`: It is an integer pointing to an index of the start of the events in the current epoch.
- `numElementsInEpoch_`: Number of events in the current epoch.

_Details on the data members_
##### `dataSeries_`
It is a vector initially created with the size of `maxEvents` and initialized with `maximum value of unsigned long integer`. `maxEvents` are maximum spikes value constructed based on epoch duration and maximum firing rate. `maxEvents` is set to 0 is there are no events from the neurons in the simulation instance, it is set to maximum spikes value otherwise.  

##### `bufferFront_`
It is the Index of the first event in the queue `dataSeries_`. It is used while checking if the vector `dataSeries_` is full.

##### `bufferEnd_`
It is an integer pointing to the location one past the end of the queue `dataSeries_`. The queue will have at-least one empty item which enables differentiation between an empty and a full queue. Since queue is the circular array implementation, `bufferEnd_` should be within valid index of the queue and we use modulus operator with the size of the queue.
That is  `bufferEnd_ = (bufferEnd_ + 1) % dataSeries_.size();`

> Initially when queue is empty  `bufferFront_` and `bufferEnd_` is 0.

> Queue full condition: `(bufferEnd_ + 1) % dataSeries_.size()) == bufferFront_` .

##### `epochStart_`
It is the index of the start of the events in the current epoch. Every-time the is `dataSeries_` is cleared, `epochStart_` is set to 0. `epochStart_` is also used to access time steps within the current epoch.

##### `numElementsInEpoch_`
It is the total number of events in the current Epoch. Ideally it is computed through `epochStart_` and `bufferEnd_`.

### Member Functions
In this section we list all the functions that are used either by vertex classes or recorder classes.
  - `EventBuffer::resize(int maxEvents)`: Initially used to size `dataSeries_` with  `maxEvents+1`, to distinguish between an empty and a full buffer. [_CAUTION: `EventBuffer` only uses this function once and resizing multiple times causes issue in the output results_]
  - `EventBuffer::operator[]`: retrieve an event time step at an offset relative to the start of the current epoch (i.e., `0..numEvents_-1`).
  - `EventBuffer::getNumElementsInEpoch()`: return the number of events in the current epoch.
  - `EventBuffer::clear()`: resets the `eventBuffer` with default values
  - `EventBuffer::insertEvent(uint64_t timeStep)`: performs enqueue operation on the buffer with event time step.
  - `EventBuffer::getPastEvent(offset: int)`: returns an event from the time in the past. `offset` determines the number of events ago the current event. `offset` must be negative.

_Details on the member functions_
##### `EventBuffer::resize(int maxEvents)`
To call this method, the current buffer must be empty. Multiple invocation might result in erroneous output. `maxEvents` are maximum spikes value constructed based on epoch duration and maximum firing rate.

##### `EventBuffer::operator[]`
This method invoked with a value between 0 and maximum number of events in the epoch. That is, element `numElementsInEpoch_ - 1` would be the last element in the epoch.

##### `EventBuffer::getNumElementsInEpoch()`
This method returns the number of events in the current epoch. Note that this might not be same as the number of events in the buffer as the enqueued events might belong to the previous epochs.

##### `EventBuffer::clear()`
This functions resets the `eventBuffer` by resetting `bufferFront_`, `bufferEnd_`,`epochStart_` and `numElementsInEpoch_` to 0.

##### `EventBuffer::insertEvent(uint64_t timeStep)`
This method checks for eventBuffer size to ensure the buffer is not full to enqueue the new event as `dataSeries_` is a circular array implementation of the queue. If the buffer `dataSeries_` is full then it is an error situation. Currently, we are not capturing errors but just making sure the event buffer is not full prior enqueuing through asserts.

##### `EventBuffer::getPastEvent(offset: int)`
This method gets the time step for an event in the past. An offset of -1 means the last event placed in the buffer; -2 means two events ago. `offset` indicates how many events ago. `offset` must be negative.
_Conditions and Observations:_
 * `Offset` must be negative.
 * `offset` must be in past, and not larger than the buffer size.
 * `bufferEnd_ + offset` would point to the desired event when the above two conditions are met.
 *  if buffer is empty `bufferFront_ == bufferEnd_`, then there are no past events events.
 * if `bufferEnd_ > bufferFront_`, then valid entries are within the queue range [bufferFront_, bufferEnd_].
 * if `bufferEnd_ < bufferFront_`, then the buffer wraps around the end of vector and valid entries are within the range [0, bufferEnd_] or in the range [bufferFront_, size()].

#### TO BE COMPLETED
Methods to copy an `EventBuffer` to and from the GPU (need to work through this; at least, it seems like this mechanism will mean that there is no need to modify the GPU code).


---------
[<< Go back to the Graphitti home page](..)
