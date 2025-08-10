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

template <typename T> class EventBuffer : public RecordableVector<T> {
public:
   /// Create EventBuffer that is sized appropriately
   ///
   /// Create and size the buffer so that it can hold a set number of events. Once an EventBuffer
   /// is created with a nonzero size, it should not be resized (generally, doing that will cause a major
   /// bug in the simulator). Note that the buffer size will be set to maxEvents+1, to distinguish between
   /// an empty and a full buffer.
   ///
   /// @param maxEvents Defaults to zero; otherwise, buffer size is set
   EventBuffer(int maxEvents = 0)
   {
      //RecordableVector<T>();
      this->dataSeries_.assign(maxEvents, numeric_limits<T>::max());
      clear();
      this->setDataType();   // set up data type for recording purpose
   }

   /** @name Recorder Interface
    *  virtual methods in RecordableBase for use by Recorder classes
    */
   ///@{

   /// Set up a string representing the basic data type
   /// Get the value of the recordable variable at the specified index.
   /// @param index The index of the recorded value to retrieve.
   /// @return A variant representing the recorded value (uint64_t, double, or string).
   variantTypes getElement(int index) const
   {
      return this->dataSeries_[(epochStart_ + index) % this->dataSeries_.size()];
   }

   /// Get the number of elements that needs to be recorded
   int getNumElements() const
   {
      return numElementsInEpoch_;
   }

   /// Start a new epoch
   ///
   /// Resets the internal variables associated with tracking the events in a epoch. Note that this doesn't
   /// affect the contents of the buffer; it just resets things so that the epoch start is the index of the next
   /// event to be enqueued and that the number of events in the epoch is 0.
   void startNewEpoch()
   {
      epochStart_ = bufferEnd_;
      bufferFront_ = bufferEnd_;
      numElementsInEpoch_ = 0;
   }
   ///@}


   /// @brief Accessor for the buffer front value.
   /// @return Returns index of the first event in the queue.
   int getBufferFront() const
   {
      return bufferFront_;
   }

   /// @brief Accessor for the buffer end value.
   /// @return Returns index of the last event in the queue.
   int getBufferEnd() const
   {
      return bufferEnd_;
   }

   /// @brief Accessor for the epoch start value.
   /// @return Returns index of the start of the events in the current epoch.
   int getEpochStart() const
   {
      return epochStart_;
   }

   /// Get number of events in the current/preceding epoch
   ///
   /// Getting the number of events in the current epoch (or, in between epochs, the number of events
   /// in the preceding epoch) is not the same as the number of events in the buffer, because the buffer
   /// retains events from the previous epoch, too.
   int getNumElementsInEpoch() const
   {
      return numElementsInEpoch_;
   }

   /// Getters are needed for copying from the GPU. Allows us to remove the friend keyword requirement.
   /// {
   /// @brief Mutator for the buffer front value.
   void setBufferFront(int bufferFront)
   {
      bufferFront_ = bufferFront;
   }

   /// @brief Mutator for the buffer end value.
   void setBufferEnd(int bufferEnd)
   {
      bufferEnd_ = bufferEnd;
   }

   /// @brief Mutator for the epoch start value.
   void setEpochStart(int epochStart)
   {
      epochStart_ = epochStart;
   }

   /// Sets number of events in the current/preceding epoch
   void setNumElementsInEpoch(int numElementsInEpoch)
   {
      numElementsInEpoch_ = numElementsInEpoch;
   }
   /// }

   /// Resize event buffer
   ///
   /// Note that the buffer size will be set to maxEvents+1, to distinguish between
   /// an empty and a full buffer.
   ///
   /// @pre current buffer must be empty
   /// @param maxEvents Buffer size
   void resize(int maxEvents)
   {
      // Only an empty buffer can be resized
      assert(this->dataSeries_.empty());
      this->dataSeries_.resize(maxEvents, 0);
      // If we resized, we should clear everything
      clear();
   }

   /// Access event from current epoch
   ///
   /// Access an element of the buffer as though it is an array or vector with element 0 being the first
   /// event in the epoch (element numElementsInEpoch_ - 1 would be the last element in the epoch).
   ///
   /// @param i element number
   T operator[](int i) const
   {
      return this->dataSeries_[(epochStart_ + i) % this->dataSeries_.size()];
   }

   /** @name Vertex and Edge Interface
    *  EventBuffer interface for use by the Vertex and Edge classes
    */
   ///@{
   /// Reset member variables consistent with an empty buffer
   void clear()
   {
      bufferFront_ = 0;
      bufferEnd_ = 0;
      epochStart_ = 0;
      numElementsInEpoch_ = 0;
   }

   /// @brief
   /// @return Returns the size of the buffer.
   int size()
   {
      return this->dataSeries_.size();
   }

   /// Insert an event time step
   ///
   /// Insert an event time step into the buffer (i.e., enqueue it in the circular array implementation of the
   /// queue).
   ///
   /// @pre The buffer is not full
   /// @param timeStep Value to store in buffer
   void insertEvent(T timeStep)
   {
      // If the buffer is full, then this is an error condition
      assert((numElementsInEpoch_ < this->dataSeries_.size()));

      // Insert time step and increment the queue end index, mod the buffer size
      this->dataSeries_[bufferEnd_] = timeStep;
      bufferEnd_ = (bufferEnd_ + 1) % this->dataSeries_.size();
      numElementsInEpoch_ += 1;
   }

   /// Get an event from a time in the past
   ///
   /// Get the time step for an event in the past. An offset of -1 means the last event placed in the
   /// buffer; -2 means two events ago.
   ///
   /// @param offset How many events ago. Must be negative. If that event isn't in the buffer,
   ///               or if the buffer is empty, returns ULONG_MAX.
   T getPastEvent(int offset) const
   {
      // Quick checks: offset must be in past, and not larger than the buffer size
      assert(((offset < 0)) && (offset > -(this->dataSeries_.size() - 1)));

      // The  event is at bufferEnd_ + offset (taking into account the
      // buffer size, and the fact that offset is negative).
      int index = bufferEnd_ + offset;
      if (index < 0)
         index += this->dataSeries_.size();

      // Need to check that we're not asking for an item so long ago that it is
      // not in the buffer. Note that there are three possibilities:
      // 1. if bufferEnd_ > bufferFront_, then valid entries are within the range
      //    [bufferFront_, bufferEnd_)
      // 2. if bufferEnd_ < bufferFront_, then the buffer wraps around the end of
      //    vector and valid entries are within the range [0, bufferEnd_) or the
      //    range [bufferFront_, size()).
      // 3. if buffer is empty (bufferFront_ == bufferEnd_), then there are no events
      //
      // Note that this means that index at this point must always be less than
      // bufferEnd_ AND >= queueFront.
      if ((index < bufferEnd_) && (index >= bufferFront_))
         return this->dataSeries_[index];
      else
         return numeric_limits<T>::max();
   }
   ///@}

   ///  Cereal serialization method
   template <class Archive> void serialize(Archive &archive)
   {
      archive(cereal::base_class<RecordableVector<T>>(this),
              cereal::make_nvp("bufferFront", bufferFront_),
              cereal::make_nvp("bufferEnd", bufferEnd_),
              cereal::make_nvp("epochStart", epochStart_),
              cereal::make_nvp("numElementsInEpoch", numElementsInEpoch_));
   }

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


// CEREAL_REGISTER_TYPE(EventBuffer<T>);
