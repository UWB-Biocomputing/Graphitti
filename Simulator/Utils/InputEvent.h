/**
 * @file InputEvent.h
 * @author Jardi A. M. Jordan (jardiamj@gmail.com)
 * @date 01-05-2023
 * Supervised by Dr. Michael Stiber, UW Bothell CSSE Division
 * @ingroup Simulator/Utils
 * 
 * @brief Structs to hold InputEvent attributes read from input files.
 * 
 * The InputEvent struct is the base class meant to hold Neural Network events
 * which are spikes that occur at a given neuron (vertex) and time.
 * 
 * The Call struct adds attributes that are specific to NG-911 events (calls).
 */


#pragma once

#include "BGTypes.h"
#include <cstdint>
#include <string>

struct InputEvent {
   // The vertexId where the input event happen
   int vertexId;
   // The start of the event since the beggining of
   // the simulation in timesteps matches g_simulationStep type
   uint64_t time;
};

struct Call : public InputEvent {
   // The duration of the event in timesteps
   int duration;
   // Event location
   BGFLOAT x;
   BGFLOAT y;
   // Patience time: How long a customer is willing to wait in the queue
   int patience;
   // On Site Time: Time spent by a responder at the site of the incident
   int onSiteTime;
   std::string type;
};