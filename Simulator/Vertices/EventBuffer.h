/**
* @file EventBuffer.h
*
* @ingroup Simulator/Vertices
*
* @brief Encapsulation of vertex event buffering
*
* An EventBuffer is a circular array based implementation of a queue of
* time steps of events produced by a vertex. It provides an interface for the
* Vertex classes, and a vector-like interface for the Recorder classes. the
* former allows event time steps to span epoch boundaries, while the latter
* provides a zero-based indexing of just the events that occurred in the
* precceding epoch.
 *
 * Currently, since there is no exception setup for the simulator, fatal errors are
 * captured using asserts.
*
* See also the eventBuffering.md file in the GitHub pages.
*
*  @author Created by Prof. Michael Stiber on 11/23/21.
*/

#pragma once
#include "Global.h"
#include "RecordableVector.h"
// cereal
#include <cereal/types/polymorphic.hpp>

class AllSpikingNeurons;
class AllIFNeurons;
class All911Vertices;
class EventBuffer : public RecordableVector<uint64_t> {
   friend class AllIFNeurons;
   friend class AllSpikingNeurons;
   friend class All911Vertices;

public:
   /// Create EventBuffer that is sized appropriately
   ///
   /// Create and size the buffer so that it can hold a set number of events. Once an EventBuffer
   /// is created with a nonzero size, it should not be resized (generally, doing that will cause a major
   /// bug in the simulator). Note that the buffer size will be set to maxEvents+1, to distinguish between
   /// an empty and a full buffer.
   ///
   /// @param maxEvents Defaults to zero; otherwise, buffer size is set
   EventBuffer(int maxEvents = 0);

   /** @name Recorder Interface
    *  virtual methods in RecordableBase for use by Recorder classes
    */
   ///@{

   /// Set up a string representing the basic data type
   /// Get the value of the recordable variable at the specified index.
   /// @param index The index of the recorded value to retrieve.
   /// @return A variant representing the recorded value (uint64_t, double, or string).
   virtual variantTypes getElement(int index) const override;

   /// Get the number of elements that needs to be recorded
   virtual int getNumElements() const override;

   /// Return the runtime data type info of unit64_t
   virtual void setDataType() override;

   /// Get the basic data type of the recordable variable
   virtual const string &getDataType() const override;

   /// Start a new epoch
   ///
   /// Resets the internal variables associated with tracking the events in a epoch. Note that this doesn't
   /// affect the contents of the buffer; it just resets things so that the epoch start is the index of the next
   /// event to be enqueued and that the number of events in the epoch is 0.
   virtual void startNewEpoch() override;
   ///@}

   /// Get number of events in the current/preceding epoch
   ///
   /// Getting the number of events in the current epoch (or, in between epochs, the number of events
   /// in the preceding epoch) is not the same as the number of events in the buffer, because the buffer
   /// retains events from the previous epoch, too.
   int getNumElementsInEpoch() const;

   /// Resize event buffer
   ///
   /// Note that the buffer size will be set to maxEvents+1, to distinguish between
   /// an empty and a full buffer.
   ///
   /// @pre current buffer must be empty
   /// @param maxEvents Buffer size
   virtual void resize(int maxEvents) override;

   /// Access event from current epoch
   ///
   /// Access an element of the buffer as though it is an array or vector with element 0 being the first
   /// event in the epoch (element numElementsInEpoch_ - 1 would be the last element in the epoch).
   ///
   /// @param i element number
   uint64_t operator[](int i) const;

   /** @name Vertex and Edge Interface
    *  EventBuffer interface for use by the Vertex and Edge classes
    */
   ///@{
   /// Reset member variables consistent with an empty buffer
   void clear();

   /// Insert an event time step
   ///
   /// Insert an event time step into the buffer (i.e., enqueue it in the circular array implementation of the
   /// queue).
   ///
   /// @pre The buffer is not full
   /// @param timeStep Value to store in buffer
   void insertEvent(uint64_t timeStep);

   /// Get an event from a time in the past
   ///
   /// Get the time step for an event in the past. An offset of -1 means the last event placed in the
   /// buffer; -2 means two events ago.
   ///
   /// @param offset How many events ago. Must be negative. If that event isn't in the buffer,
   ///               or if the buffer is empty, returns ULONG_MAX.
   uint64_t getPastEvent(int offset) const;
   ///@}

   ///  Cereal serialization method
   template <class Archive> void serialize(Archive &archive);

private:
   /// Holds the event time steps
   // vector<uint64_t> dataSeries_;

   /// Index of the first event in the queue
   int bufferFront_;

   /// Index of the location one past the end of the queue; where the next event will be enqueued. Note
   /// that the array must always have one empty item; otherwise, it would not be possible to tell the
   /// difference between an empty and a full queue. Specific cases:
   /// Case | bufferFront_ | bufferEnd_
   /// --- | --- | ---
   /// Initial (empty) queue | 0 | 0
   /// empty queue (otherwise) | i | i
   /// non-empty queue | i | (i + offset) % dataSeries_.size()
   /// full queue | i | (i - 1) (dataSeries_.size() - 1 if i==0)
   int bufferEnd_;

   /// Index of the start of the events in the current epoch
   int epochStart_;

   /// Number of events in the current epoch. Note that this could be computed from epochStart_
   /// and bufferEnd_, but the code to do that would be unobvious.
   int numElementsInEpoch_;
};


CEREAL_REGISTER_TYPE(EventBuffer);

///  Cereal serialization method
template <class Archive> void EventBuffer::serialize(Archive &archive)
{
   archive(cereal::base_class<RecordableVector<uint64_t>>(this),
           cereal::make_nvp("bufferFront", bufferFront_), cereal::make_nvp("bufferEnd", bufferEnd_),
           cereal::make_nvp("epochStart", epochStart_),
           cereal::make_nvp("numElementsInEpoch", numElementsInEpoch_));
}
