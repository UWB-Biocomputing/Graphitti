/**
 * @file Serializer.h
 *
 * @ingroup Simulator/Core
 * 
 * @brief Handles implementation of serialization and deserialization of synapses.
 * 
 *  Serializes and Deserializes synapse weights, source vertices, destination vertices,
 *  maxEdgesPerVertex, totalVertices and radii.
 */
#pragma once

class Serializer {
public:
   Serializer() = default;

   ///  Serializes synapse weights, source vertices, destination vertices,
   ///  maxEdgesPerVertex, totalVertices.
   ///  if running a connGrowth model serializes radii as well
   void serializeSynapses();

   ///  Deserializes synapse weights, source vertices, destination vertices,
   ///  maxEdgesPerVertex, totalVertices.
   ///  if running a connGrowth model and radii is in serialization file, deserializes radii as well
   ///
   ///  @returns    true if successful, false otherwise.
   bool deserializeSynapses();
};
