/**
 * @file Serializer.h
 *
 * @ingroup Simulator/Core
 * 
 * @brief Provides serialization and deserialization functionality using the Cereal library.
 * 
 * This class handles the serialization and deserialization of all member variables 
 * in the Connections, Layout, Edges, Vertices, and associated helper classes such as
 * EdgeIndexMap, Model, RecordableBase, RecordableVector, Matrix, RNG and EventBuffer.
 * Note that Recorder class is not serialized or deserialized.
 * 
 * The serialization and deserialization process typically begins with the Model class,
 * which internally calls the serialization of the Connections and Layout classes.
 * Connections, in turn, handle the serialization of Edges, while Layout handles 
 * the serialization of Vertices. This ensures a comprehensive serialization of 
 * the entire simulation structure.
 * 
 * @note As of September 2024, serialization support is currently available 
 * for CPU-based Neuron simulations. While GPU-based Neuron serialization is functional, 
 * the output result files differ, and this is being addressed in [Issue #701].
 * Serialization support for NG911 will be extended in the future [Issue #700].
 *  
 */

#pragma once
#include "Simulator.h"

class Serializer {
public:
   Serializer() = default;

   /// Serializes all member variables of the
   /// Connections, Layout, Edges, Vertices, and associated helper classes.
   void serialize();

   /// Deserializes all member variables of the
   /// Connections, Layout, Edges, Vertices, and associated helper classes.
   ///
   /// @returns true if deserialization is successful; false otherwise.
   bool deserialize();

private:
   template <typename Archive> static bool processArchive(Archive &archive, Simulator &simulator);
};
