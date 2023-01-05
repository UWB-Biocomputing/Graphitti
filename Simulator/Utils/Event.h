/**
 * @file Event.h
 * @author Jardi A. M. Jordan (jardiamj@gmail.com)
 * @date 01-05-2023
 * Supervised by Dr. Michael Stiber, UW Bothell CSSE Division
 * @ingroup Simulator/Utils
 * 
 * @brief Structs to hold Event attributes read from input files.
 * 
 * The Event struct is the base class meant to hold Neural Network events
 * which are spikes that occur at a given neuron (vertex) and time.
 * 
 * The Call struct adds attributes that are specific to NG-911 events (calls).
 */


#pragma once

struct Event {
   // The vertexId where the input event happen
   int vertexId;
   // The start of the event since the beggining of
   // the simulation in timesteps matches g_simulationStep type
   uint64_t time;
};

struct Call : Event {
    // The duration of the event in timesteps
   int duration;
   // Event location
   double x;
   double y;
   string type;
};