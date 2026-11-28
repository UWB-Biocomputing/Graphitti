/**
 * @file CallUtils.h
 * @ingroup Simulator/Utils
 *
 * @brief Helpers for NG911 call field vectors and responder type conversion.
 */

#pragma once

#include "BGTypes.h"
#include "InputEvent.h"
#include "VertexType.h"
#include <cstdint>
#include <string>

/// Responder type codes matching vertexType enum values for EMS, FIRE, and LAW.
inline int responderTypeToInt(const std::string &type)
{
   if (type == "Law") {
      return static_cast<int>(vertexType::LAW);
   }
   if (type == "EMS") {
      return static_cast<int>(vertexType::EMS);
   }
   if (type == "Fire") {
      return static_cast<int>(vertexType::FIRE);
   }
   return static_cast<int>(vertexType::VTYPE_UNDEF);
}

inline std::string responderTypeToString(int responderType)
{
   if (responderType == static_cast<int>(vertexType::LAW)) {
      return "Law";
   }
   if (responderType == static_cast<int>(vertexType::EMS)) {
      return "EMS";
   }
   if (responderType == static_cast<int>(vertexType::FIRE)) {
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
