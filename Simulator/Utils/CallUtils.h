/**
 * @file CallUtils.h
 * @ingroup Simulator/Utils
 *
 * @brief Helpers for NG911 call field vectors and responder type conversion.
 */

#pragma once

#include "BGTypes.h"
#include "InputEvent.h"
#include <cstdint>
#include <string>

/// Responder type codes matching vertexType enum values for EMS, FIRE, and LAW.
inline int responderTypeToInt(const std::string &type)
{
   if (type == "Law") {
      return 7;
   }
   if (type == "EMS") {
      return 5;
   }
   if (type == "Fire") {
      return 6;
   }
   return 0;
}

inline std::string responderTypeToString(int responderType)
{
   if (responderType == 7) {
      return "Law";
   }
   if (responderType == 5) {
      return "EMS";
   }
   if (responderType == 6) {
      return "Fire";
   }
   return "";
}

inline Call makeCall(int vertexId, uint64_t time, int duration, BGFLOAT x, BGFLOAT y, int patience,
                     int onSiteTime, int responderType)
{
   Call call;
   call.vertexId = vertexId;
   call.time = time;
   call.duration = duration;
   call.x = x;
   call.y = y;
   call.patience = patience;
   call.onSiteTime = onSiteTime;
   call.type = responderTypeToString(responderType);
   return call;
}

inline Call makeCall(const Call &call, int responderTypeOverride = -1)
{
   if (responderTypeOverride >= 0) {
      return makeCall(call.vertexId, call.time, call.duration, call.x, call.y, call.patience,
                      call.onSiteTime, responderTypeOverride);
   }
   return makeCall(call.vertexId, call.time, call.duration, call.x, call.y, call.patience,
                   call.onSiteTime, responderTypeToInt(call.type));
}
