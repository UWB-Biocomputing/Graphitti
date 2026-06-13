/**
 * @file CallCircularBuffer.h
 * @ingroup Simulator/Utils
 *
 * @brief Circular buffer for NG911 calls using separate field vectors.
 *
 * Mirrors the GPU layout so host and device memory can be copied directly.
 */

#pragma once

#include "BGTypes.h"
#include "CallUtils.h"
#include "InputEvent.h"
#include <cassert>
#include <cstddef>
#include <optional>
#include <vector>

class CallCircularBuffer {
public:
   explicit CallCircularBuffer(int capacity = 0)
   {
      resize(capacity);
      clear();
   }

   void resize(int capacity)
   {
      assert(isEmpty());
      const size_t bufferSize = static_cast<size_t>(capacity) + 1;
      vertexId_.resize(bufferSize);
      time_.resize(bufferSize);
      duration_.resize(bufferSize);
      x_.resize(bufferSize);
      y_.resize(bufferSize);
      patience_.resize(bufferSize);
      onSiteTime_.resize(bufferSize);
      responderType_.resize(bufferSize);
      clear();
   }

   void put(const Call &call)
   {
      assert(!isFull());
      setAt(front_, call);
      front_ = (front_ + 1) % bufferSize();
   }

   std::optional<Call> get()
   {
      if (isEmpty()) {
         return std::nullopt;
      }

      Call value = callAt(end_);
      end_ = (end_ + 1) % bufferSize();
      return value;
   }

   std::optional<Call> peek() const
   {
      if (isEmpty()) {
         return std::nullopt;
      }
      return callAt(end_);
   }

   void clear()
   {
      front_ = 0;
      end_ = 0;
   }

   bool isEmpty() const
   {
      return front_ == end_;
   }

   bool isFull() const
   {
      return ((front_ + 1) % bufferSize()) == end_;
   }

   size_t capacity() const
   {
      return bufferSize() - 1;
   }

   size_t size() const
   {
      if (front_ >= end_) {
         return front_ - end_;
      }
      return bufferSize() + front_ - end_;
   }

   size_t getFrontIndex() const
   {
      return front_;
   }

   size_t getEndIndex() const
   {
      return end_;
   }

   void setFrontIndex(unsigned long front)
   {
      front_ = front;
   }

   void setEndIndex(unsigned long end)
   {
      end_ = end;
   }

   size_t bufferSize() const
   {
      return vertexId_.size();
   }

   void resizeBuffer(size_t bufferSize)
   {
      vertexId_.resize(bufferSize);
      time_.resize(bufferSize);
      duration_.resize(bufferSize);
      x_.resize(bufferSize);
      y_.resize(bufferSize);
      patience_.resize(bufferSize);
      onSiteTime_.resize(bufferSize);
      responderType_.resize(bufferSize);
   }

   Call callAt(size_t index) const
   {
      return makeCall(vertexId_[index], time_[index], duration_[index], x_[index], y_[index],
                      patience_[index], onSiteTime_[index], responderType_[index]);
   }

   void setAt(size_t index, const Call &call)
   {
      vertexId_[index] = call.vertexId;
      time_[index] = call.time;
      duration_[index] = call.duration;
      x_[index] = call.x;
      y_[index] = call.y;
      patience_[index] = call.patience;
      onSiteTime_[index] = call.onSiteTime;
      responderType_[index] = responderTypeToInt(call.type);
   }

   void copyAt(size_t dstIndex, size_t srcIndex)
   {
      vertexId_[dstIndex] = vertexId_[srcIndex];
      time_[dstIndex] = time_[srcIndex];
      duration_[dstIndex] = duration_[srcIndex];
      x_[dstIndex] = x_[srcIndex];
      y_[dstIndex] = y_[srcIndex];
      patience_[dstIndex] = patience_[srcIndex];
      onSiteTime_[dstIndex] = onSiteTime_[srcIndex];
      responderType_[dstIndex] = responderType_[srcIndex];
   }

   std::vector<int> &vertexId()
   {
      return vertexId_;
   }

   std::vector<uint64_t> &time()
   {
      return time_;
   }

   std::vector<int> &duration()
   {
      return duration_;
   }

   std::vector<BGFLOAT> &x()
   {
      return x_;
   }

   std::vector<BGFLOAT> &y()
   {
      return y_;
   }

   std::vector<int> &patience()
   {
      return patience_;
   }

   std::vector<int> &onSiteTime()
   {
      return onSiteTime_;
   }

   std::vector<int> &responderType()
   {
      return responderType_;
   }

   const std::vector<int> &vertexId() const
   {
      return vertexId_;
   }

   const std::vector<uint64_t> &time() const
   {
      return time_;
   }

   const std::vector<int> &duration() const
   {
      return duration_;
   }

   const std::vector<BGFLOAT> &x() const
   {
      return x_;
   }

   const std::vector<BGFLOAT> &y() const
   {
      return y_;
   }

   const std::vector<int> &patience() const
   {
      return patience_;
   }

   const std::vector<int> &onSiteTime() const
   {
      return onSiteTime_;
   }

   const std::vector<int> &responderType() const
   {
      return responderType_;
   }

private:
   std::vector<int> vertexId_;
   std::vector<uint64_t> time_;
   std::vector<int> duration_;
   std::vector<BGFLOAT> x_;
   std::vector<BGFLOAT> y_;
   std::vector<int> patience_;
   std::vector<int> onSiteTime_;
   std::vector<int> responderType_;
   size_t front_ = 0;
   size_t end_ = 0;
};
