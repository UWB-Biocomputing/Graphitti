## CPU GPU Architecture

Graphitti is a high-performance simulator of graph-based systems, currently being applied to computational neuroscience and emergency communication systems. It runs on both CPUs and GPUs and can simulate very large graphs (tens of thousands of vertices; hundreds of thousands to millions of edges) for long durations (billions of time steps).

The typical process for implementing a new system using Graphitti is to implement the system on the CPU and then build a corresponding GPU implementation.



When creating the GPU implementation, we take care to implement the logic so that it easily maps to the CPU implementation. For example, we have 3 separate methods in the CPU All911Vertices class defining how a vertex should advance depending on the type of vertex that it is (caller region, PSAP, or emergency responder). In the GPU All911Vertices class, we mirror this by defining 3 separate kernel methods that hold similar logic, but work on the corresponding Device data members.

CPU implementation of advanceCALR method for advancing a caller region vertex:
```cpp
void All911Vertices::advanceCALR(BGSIZE vertexIdx, All911Edges &edges911,
                                 const EdgeIndexMap &edgeIndexMap)
{
   // There is only one outgoing edge from CALR to a PSAP
   BGSIZE start = edgeIndexMap.outgoingEdgeBegin_[vertexIdx];
   BGSIZE edgeIdx = edgeIndexMap.outgoingEdgeIndexMap_[start];

   // Check for dropped calls, indicated by the edge not being available
   if (!edges911.isAvailable_[edgeIdx]) {
      // If the call is still there, it means that there was no space in the PSAP's waiting
      // queue. Therefore, this is a dropped call.
      // If readialing, we assume that it happens immediately and the caller tries until
      // getting through.
      if (!edges911.isRedial_[edgeIdx] && initRNG.randDblExc() >= redialP_) {
         // We only make the edge available if no readialing occurs.
         edges911.isAvailable_[edgeIdx] = true;
         LOG4CPLUS_DEBUG(vertexLogger_, "Did not redial at time: " << edges911.call_[edgeIdx].time);
      } else {
         // Keep the edge unavailable but mark it as a redial
         edges911.isRedial_[edgeIdx] = true;
      }
   }

   // peek at the next call in the queue
   optional<Call> nextCall = vertexQueues_[vertexIdx].peek();
   if (edges911.isAvailable_[edgeIdx] && nextCall && nextCall->time <= g_simulationStep) {
      // Calls that start at the same time are process in the order they appear.
      // The call starts at the current time step so we need to pop it and process it
      vertexQueues_[vertexIdx].get();   // pop from the queue

      // Place new call in the edge going to the PSAP
      assert(edges911.isAvailable_[edgeIdx]);
      edges911.call_[edgeIdx] = nextCall.value();
      edges911.isAvailable_[edgeIdx] = false;
      LOG4CPLUS_DEBUG(vertexLogger_, "Calling PSAP at time: " << nextCall->time);
   }
}
```

Corresponding GPU kernel method for advancing a caller region vertex:
```cpp
CUDA_CALLABLE void advanceCALRVerticesDevice(int vertexId,
                                             int totalNumberOfEvents,
                                             uint64_t simulationStep,
                                             BGFLOAT redialProbability, 
                                             All911EdgesDeviceProperties *allEdgesDevice, 
                                             EdgeIndexMapDevice *edgeIndexMapDevice)
{
   // There is only one outgoing edge from CALR to a PSAP
   BGSIZE start = edgeIndexMapDevice->outgoingEdgeBegin_[vertexId];
   BGSIZE edgeIdx = edgeIndexMapDevice->outgoingEdgeIndexMap_[start];

   // Check for dropped calls, indicated by the edge not being available
   if (!allEdgesDevice->isAvailable_[edgeIdx]) {
      // If the call is still there, it means that there was no space in the PSAP's waiting
      // queue. Therefore, this is a dropped call.
      // If readialing, we assume that it happens immediately and the caller tries until
      // getting through.
      if (!allEdgesDevice->isRedial_[edgeIdx] && initRNG.randDblExc() >= redialProbability) {
         // We only make the edge available if no readialing occurs.
         allEdgesDevice->isAvailable_[edgeIdx] = true;
         //LOG4CPLUS_DEBUG(vertexLogger_, "Did not redial at time: " << edges911.call_[edgeIdx].time);
      } else {
         // Keep the edge unavailable but mark it as a redial
         allEdgesDevice->isRedial_[edgeIdx] = true;
      }
   }

   // peek at the next call in the queue
   uint64_t queueEndIndex = allVerticesDevice->vertexQueuesEnd_[vertexId];
   if (allEdgesDevice->isAvailable_[edgeIdx] && 
      (allVerticesDevice->vertexQueuesFront_[vertexId] != queueEndIndex) && 
      allVerticesDevice->vertexQueuesBufferTime_[vertexId][queueEndIndex] <= simulationStep) {
      // Place new call in the edge going to the PSAP
      assert(allEdgesDevice->isAvailable_[edgeIdx]);
      // Calls that start at the same time are process in the order they appear.
      // The call starts at the current time step so we need to pop it and process it

      // Process the call
      allEdgesDevice->vertexId_[edgeIdx] = allVerticesDevice->vertexQueuesBufferVertexId_[vertexId][queueEndIndex];
      allEdgesDevice->time_[edgeIdx] = allVerticesDevice->vertexQueuesBufferTime_[vertexId][queueEndIndex];
      allEdgesDevice->duration_[edgeIdx] = allVerticesDevice->vertexQueuesBufferDuration_[vertexId][queueEndIndex];
      allEdgesDevice->x_[edgeIdx] = allVerticesDevice->vertexQueuesBufferX_[vertexId][queueEndIndex];
      allEdgesDevice->y_[edgeIdx] = allVerticesDevice->vertexQueuesBufferY_[vertexId][queueEndIndex];
      allEdgesDevice->patience_[edgeIdx] = allVerticesDevice->vertexQueuesBufferPatience_[vertexId][queueEndIndex];
      allEdgesDevice->onSiteTime_[edgeIdx] = allVerticesDevice->vertexQueuesBufferOnSiteTime_[vertexId][queueEndIndex];
      allEdgesDevice->responderType_[edgeIdx] = allVerticesDevice->vertexQueuesBufferResponderType_[vertexId][queueEndIndex];

      // Pop from the queue
      allVerticesDevice->vertexQueuesEnd_[vertexId] = (queueEndIndex + 1) % totalNumberOfEvents;
      allEdgesDevice->isAvailable_[edgeIdx] = false;
      //LOG4CPLUS_DEBUG(vertexLogger_, "Calling PSAP at time: " << nextCall->time);
   }
}
```
In the case where a data structure isn't implemented in CUDA, we expand out the CPU implementation and directly implement this on the GPU.

Transfering a call in the integrateVertexInputs method on the CPU involves calling a Queue.Put() method:
```cpp
// Transfer call to destination
dstQueue.put(all911Edges.call_[edgeIdx]);
```

dstQueue is of type CircularBuffer defined in the CircularBuffer.h:
```cpp
void put(T element)
{
    // We throw an error if the buffer is full
    assert(!isFull());

    // Insert the new element and increment the front index
    buffer_[front_] = element;
    front_ = (front_ + 1) % buffer_.size();
}

bool isFull() const
{
    return ((front_ + 1) % buffer_.size()) == end_;
}
```

When we expand it and implement on the GPU, we get an implementation like below:
```cpp
uint64_t queueFrontIndex = allVerticesDevice->vertexQueuesFront_[dstIndex];
uint64_t queueEndIndex = allVerticesDevice->vertexQueuesEnd_[dstIndex];

// Transfer call to destination
// We throw an error if the buffer is full
assert(!(((queueFrontIndex + 1) % totalNumberOfEvents) == queueEndIndex));

// Insert the new element and increment the front index
allVerticesDevice->vertexQueuesBufferVertexId_[dstIndex][queueFrontIndex] = allEdgesDevice->vertexId_[edgeIdx];
allVerticesDevice->vertexQueuesBufferTime_[dstIndex][queueFrontIndex] = allEdgesDevice->time_[edgeIdx];
allVerticesDevice->vertexQueuesBufferDuration_[dstIndex][queueFrontIndex] = allEdgesDevice->duration_[edgeIdx];
allVerticesDevice->vertexQueuesBufferX_[dstIndex][queueFrontIndex] = allEdgesDevice->x_[edgeIdx];
allVerticesDevice->vertexQueuesBufferY_[dstIndex][queueFrontIndex] = allEdgesDevice->y_[edgeIdx];
allVerticesDevice->vertexQueuesBufferPatience_[dstIndex][queueFrontIndex] = allEdgesDevice->patience_[edgeIdx];
allVerticesDevice->vertexQueuesBufferOnSiteTime_[dstIndex][queueFrontIndex] = allEdgesDevice->onSiteTime_[edgeIdx];
allVerticesDevice->vertexQueuesBufferResponderType_[dstIndex][queueFrontIndex] = allEdgesDevice->responderType_[edgeIdx];

allVerticesDevice->vertexQueuesFront_[dstIndex] = (queueFrontIndex + 1) % totalNumberOfEvents;
```
where the size of each dstQueue is the total number of events to be run in the simulator.