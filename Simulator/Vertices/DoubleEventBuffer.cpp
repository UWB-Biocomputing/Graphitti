/**
*  @file DoubleEventBuffer.cpp
*
* @ingroup Simulator/Vertices
*
* @brief Encapsulation of vertex event buffering
*/

#include "DoubleEventBuffer.h"
#include "Global.h"
#include <cassert>
#include <limits>

DoubleEventBuffer::DoubleEventBuffer(int maxEvents)
{
   dataSeries_.assign(maxEvents, numeric_limits<float>::max());
   clear();
   setDataType();   // set up data type for recording purpose
}


// set up a string representing the basic data type
void DoubleEventBuffer::setDataType()
{
   basicDataType_ = typeid(float).name();
}

/// @brief Get the value of the recordable variable at the specified index.
/// @param index The index of the recorded value to retrieve.
/// @return A variant representing the recorded value (uint64_t, double, or string).
variantTypes DoubleEventBuffer::getElement(int index) const
{
   return dataSeries_[(epochStart_ + index) % dataSeries_.size()];
}


const string &DoubleEventBuffer::getDataType() const
{
   return basicDataType_;
}

int DoubleEventBuffer::getNumElementsInEpoch() const
{
   return numElementsInEpoch_;
}

int DoubleEventBuffer::getNumElements() const
{
   return numElementsInEpoch_;
}

void DoubleEventBuffer::resize(int maxEvents)
{
   // Only an empty buffer can be resized
   assert(dataSeries_.empty());
   dataSeries_.resize(maxEvents, 0);
   // If we resized, we should clear everything
   clear();
}

void DoubleEventBuffer::clear()
{
   bufferFront_ = 0;
   bufferEnd_ = 0;
   epochStart_ = 0;
   numElementsInEpoch_ = 0;
}

float DoubleEventBuffer::operator[](int i) const
{
   return dataSeries_[(epochStart_ + i) % dataSeries_.size()];
}

void DoubleEventBuffer::startNewEpoch()
{
   epochStart_ = bufferEnd_;
   bufferFront_ = bufferEnd_;
   numElementsInEpoch_ = 0;
}

void DoubleEventBuffer::insertEvent(float timeStep)
{
   // If the buffer is full, then this is an error condition
   assert((numElementsInEpoch_ < dataSeries_.size()));

   // Insert time step and increment the queue end index, mod the buffer size
   dataSeries_[bufferEnd_] = timeStep;
   bufferEnd_ = (bufferEnd_ + 1) % dataSeries_.size();
   numElementsInEpoch_ += 1;
}

float DoubleEventBuffer::getPastEvent(int offset) const
{
   // Quick checks: offset must be in past, and not larger than the buffer size
   assert(((offset < 0)) && (offset > -(dataSeries_.size() - 1)));

   // The  event is at bufferEnd_ + offset (taking into account the
   // buffer size, and the fact that offset is negative).
   int index = bufferEnd_ + offset;
   if (index < 0)
      index += dataSeries_.size();

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
      return dataSeries_[index];
   else
      return numeric_limits<float>::max();
}