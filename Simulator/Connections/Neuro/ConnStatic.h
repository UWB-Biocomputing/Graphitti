/**
 * @file ConnStatic.h
 *
 * @ingroup Simulator/Connections
 * 
 * @brief The model of the small world network
 *
 * The small-world networks are regular networks rewired to introduce increasing amounts
 * of disorder, which can be highly clustered, like regular lattices, yet have small
 * characterisic path length, like random graphs. 
 *
 * The structural properties of these graphs are quantified by their characteristic path
 * length \f$L(p)\f$ and clustering coefficient \f$C(p)\f$. Here \f$L\f$ is defined as the number of edges
 * in the shortest path between two vertices, average over all pairs of vertices.
 * The clustering coefficient \f$C(p)\f$ is defined as follows. Suppose that a vertex \f$v\f$ has \f$k_v\f$
 * neighbours; then at most \f$k_v (k_v - 1) / 2\f$ edges can exist between them (this occurs when
 * every neighbour of \f$v\f$ is connected to every other neighbour of \f$v\f$).
 * Let \f$C_v\f$ denote the fracion of these allowable edges that actually exist.
 * Define \f$C\f$ as the avarage of \f$C_v\f$ over all \f$v\f$ (Watts etal. 1998).
 *
 * We first create a regular network characterised by two parameters: number of maximum 
 * connections per neurons and connection radius threshold, then rewire it according 
 * to the small-world rewiring probability.
 */

#pragma once

#include "Global.h"
#include "Connections.h"
#include "Simulator.h"
#include <vector>
#include <iostream>
// cereal
#include <cereal/types/vector.hpp>

using namespace std;

class ConnStatic : public Connections {
public:
   ConnStatic();

   virtual ~ConnStatic();

   static Connections *Create() { return new ConnStatic(); }

   ///  Setup the internal structure of the class (allocate memories and initialize them).
   ///  Initialize the small world network characterized by parameters:
   ///  number of maximum connections per vertex, connection radius threshold, and
   ///  small-world rewiring probability.
   ///
   ///  @param  layout    Layout information of the neural network.
   ///  @param  vertices   The Neuron list to search from.
   ///  @param  edges  The Synapse list to search from.
   virtual void setupConnections(Layout *layout, AllVertices *vertices, AllEdges *edges) override;

   /// Load member variables from configuration file.
   /// Registered to OperationManager as Operations::op::loadParameters
   virtual void loadParameters() override;

   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const override;

   ///  Get connection radius threshold
   BGFLOAT getConnsRadiusThresh() const { return threshConnsRadius_; }

   /// Get array of vertex weights
   BGFLOAT* getWCurrentEpoch() const { return WCurrentEpoch_; }

   /// Get all edge source vertex indices
   int* getSourceVertexIndexCurrentEpoch() const { return sourceVertexIndexCurrentEpoch_; }

      /// Get all edge destination vertex indices
   int* getDestVertexIndexCurrentEpoch() const { return destVertexIndexCurrentEpoch_; }

  ///  Cereal serialization method
  ///  (Serializes radii)
   template<class Archive>
   void save(Archive &archive) const;
   
   ///  Cereal deserialization method
   ///  (Deserializes radii)
   template<class Archive>
   void load(Archive &archive);
   
private:
   /// Indices of the source vertex for each edge
   int *sourceVertexIndexCurrentEpoch_;
   
    /// Indices of the destination vertex for each edge
   int *destVertexIndexCurrentEpoch_;
   
    /// The weight (scaling factor, strength, maximal amplitude) of each vertex for the current epoch.
   BGFLOAT *WCurrentEpoch_;
   
   /// radii size （2020/2/13 add radiiSize for use in serialization/deserialization)
   int radiiSize_;

   /// number of maximum connections per vertex
   int connsPerVertex_;

   /// Connection radius threshold
   BGFLOAT threshConnsRadius_;

   /// Small-world rewiring probability
   BGFLOAT rewiringProbability_;

   /// Min/max values of excitatory neuron's synapse weight
   BGFLOAT excWeight_[2];

   /// Min/max values of inhibitory neuron's synapse weight
   BGFLOAT inhWeight_[2];

   struct DistDestVertex {
      BGFLOAT dist;     ///< destance to the destination vertex
      int destVertex;  ///< index of the destination vertex

      bool operator<(const DistDestVertex &other) const {
         return (dist < other.dist);
      }
   };
};
