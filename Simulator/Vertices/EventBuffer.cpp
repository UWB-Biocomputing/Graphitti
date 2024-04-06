/**
*  @file EventBuffer.cpp
*
* @ingroup Simulator/Vertices
 *
 * @brief Encapsulation of vertex event buffering
*
* @author Created by Prof. Michael Stiber on 11/23/21.
*/

#include "EventBuffer.h"
#include "Global.h"
#include <cassert>
#include <limits>

// EventBuffer::EventBuffer(int maxEvents) :
//    eventTimeSteps_(maxEvents + 1, numeric_limits<unsigned long>::max())
// {
//    clear();
// }
EventBuffer::EventBuffer(int maxEvents)
{
   eventTimeSteps_.assign(maxEvents + 1, numeric_limits<unsigned long>::max());
   clear();
   setDataType();   // set up data type for recording purpose
}


// set up a string representing the basic data type
void EventBuffer::setDataType()
{
   basicDataType_ = typeid(uint64_t).name();
}

/// @brief Get the value of the recordable variable at the specified index.
/// @param index The index of the recorded value to retrieve.
/// @return A variant representing the recorded value (uint64_t, double, or string).
variantTypes EventBuffer::getElement(int index) const
{
   return eventTimeSteps_[(epochStart_ + index) % eventTimeSteps_.size()];
   // return eventTimeSteps_[index];
}


const string &EventBuffer::getDataType() const
{
   return basicDataType_;
}

void EventBuffer::startNewEpoch()
{
   epochStart_ = queueEnd_;
   numEventsInEpoch_ = 0;
}

int EventBuffer::getNumElements() const
{
   return getNumEventsInEpoch();
}

int EventBuffer::getNumEventsInEpoch() const
{
   return numEventsInEpoch_;
}

void EventBuffer::resize(int maxEvents)
{
   // Only an empty buffer can be resized
   assert(eventTimeSteps_.empty());
   eventTimeSteps_.resize(maxEvents + 1, 0);
   // If we resized, we should clear everything
   clear();
}

void EventBuffer::clear()
{
   queueFront_ = 0;
   queueEnd_ = 0;
   epochStart_ = 0;
   numEventsInEpoch_ = 0;
}

uint64_t EventBuffer::operator[](int i) const
{
   return eventTimeSteps_[(epochStart_ + i) % eventTimeSteps_.size()];
}


void EventBuffer::insertEvent(uint64_t timeStep)
{
   // If the buffer is full, then this is an error condition
   assert(((queueEnd_ + 1) % eventTimeSteps_.size()) != queueFront_);

   // Insert time step and increment the queue end index, mod the buffer size
   eventTimeSteps_[queueEnd_] = timeStep;
   queueEnd_ = (queueEnd_ + 1) % eventTimeSteps_.size();
   numEventsInEpoch_ += 1;
}

uint64_t EventBuffer::getPastEvent(int offset) const
{
   // Quick checks: offset must be in past, and not larger than the buffer size
   assert(((offset < 0)) && (offset > -(eventTimeSteps_.size() - 1)));

   // The  event is at queueEnd_ + offset (taking into account the
   // buffer size, and the fact that offset is negative).
   int index = queueEnd_ + offset;
   if (index < 0)
      index += eventTimeSteps_.size();

   // Need to check that we're not asking for an item so long ago that it is
   // not in the buffer. Note that there are three possibilities:
   // 1. if queueEnd_ > queueFront_, then valid entries are within the range
   //    [queueFront_, queueEnd_)
   // 2. if queueEnd_ < queueFront_, then the buffer wraps around the end of
   //    vector and valid entries are within the range [0, queueEnd_) or the
   //    range [queueFront_, size()).
   // 3. if buffer is empty (queueFront_ == queueEnd_), then there are no events
   //
   // Note that this means that index at this point must always be less than
   // queueEnd_ AND >= queueFront.
   if ((index < queueEnd_) && (index >= queueFront_))
      return eventTimeSteps_[index];
   else
      return numeric_limits<unsigned long>::max();
}
