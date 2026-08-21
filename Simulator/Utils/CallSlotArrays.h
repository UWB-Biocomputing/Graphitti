/**
 * @file CallSlotArrays.h
 * @ingroup Simulator/Utils
 *
 * @brief Fixed-size per-vertex call storage using separate field vectors.
 */

#pragma once

#include "BGTypes.h"
#include "CallUtils.h"
#include "InputEvent.h"
#include <vector>

class CallSlotArrays {
public:
   void resize(size_t numSlots)
   {
      vertexId_.assign(numSlots, 0);
      time_.assign(numSlots, 0);
      duration_.assign(numSlots, 0);
      x_.assign(numSlots, 0);
      y_.assign(numSlots, 0);
      patience_.assign(numSlots, 0);
      onSiteTime_.assign(numSlots, 0);
      responderType_.assign(numSlots, 0);
   }

   size_t size() const
   {
      return vertexId_.size();
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

   void setTimeAt(size_t index, uint64_t time)
   {
      time_[index] = time;
   }

   uint64_t timeAt(size_t index) const
   {
      return time_[index];
   }

   int responderTypeAt(size_t index) const
   {
      return responderType_[index];
   }

   BGFLOAT xAt(size_t index) const
   {
      return x_[index];
   }

   BGFLOAT yAt(size_t index) const
   {
      return y_[index];
   }

   int patienceAt(size_t index) const
   {
      return patience_[index];
   }

   int onSiteTimeAt(size_t index) const
   {
      return onSiteTime_[index];
   }

   int durationAt(size_t index) const
   {
      return duration_[index];
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
};
