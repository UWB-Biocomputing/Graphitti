
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