/**
 * @file ConnStatic.h
 *
 * @ingroup Simulator/Connections
 * 
 * @brief The model of the small world network
 *
 * The small-world networks are regular networks rewired to introduce increasing amounts
 * of disorder, which can be highly clustered, like regular lattices, yet have small
 * characteristic path length, like random graphs. 
 *
 * The structural properties of these graphs are quantified by their characteristic path
 * length \f$L(p)\f$ and clustering coefficient \f$C(p)\f$. Here \f$L\f$ is defined as the number of edges
 * in the shortest path between two vertices, average over all pairs of vertices.
 * The clustering coefficient \f$C(p)\f$ is defined as follows. Suppose that a vertex \f$v\f$ has \f$k_v\f$
 * neighbors; then at most \f$k_v (k_v - 1) / 2\f$ edges can exist between them (this occurs when
 * every neighbor of \f$v\f$ is connected to every other neighbor of \f$v\f$).
 * Let \f$C_v\f$ denote the fracion of these allowable edges that actually exist.
 * Define \f$C\f$ as the average of \f$C_v\f$ overall \f$v\f$ (Watts et al. 1998).
 */

#pragma once

#include "Connections.h"
#include "Global.h"
#include "RecordableVector.h"
#include "Simulator.h"
#include <iostream>
#include <vector>
// cereal
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>

using namespace std;

class ConnStatic : public Connections {
public:
   ConnStatic();

   virtual ~ConnStatic() = default;

   static Connections *Create()
   {
      return new ConnStatic();
   }

   /// Register vertex properties with the GraphManager
   virtual void registerGraphProperties() override;

   ///  Setup the internal structure of the class (allocate memories and initialize them).
   ///  Initialize the small world network characterized by parameters:
   ///  number of maximum connections per vertex, connection radius threshold, and
   ///  small-world rewiring probability.
   virtual void setup() override;

   /// Load member variables from configuration file.
   /// Registered to OperationManager as Operations::op::loadParameters
   virtual void loadParameters() override;

   ///  Prints out all parameters to the logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const override;

   /// Registers history variables for recording during simulation 
   virtual void registerHistoryVariables() override;

   /// Get array of vertex weights
   const vector<BGFLOAT> &getWCurrentEpoch() const
   {
      // return WCurrentEpoch_;
      return WCurrentEpoch_.getVector();
   }

   /// Get all edge source vertex indices
   const vector<int> &getSourceVertexIndexCurrentEpoch() const
   {
      return sourceVertexIndexCurrentEpoch_.getVector();
   }

   /// Get all edge destination vertex indices
   const vector<int> &getDestVertexIndexCurrentEpoch() const
   {
      return destVertexIndexCurrentEpoch_.getVector();
   }

   ///  Cereal serialization method
   template <class Archive> void serialize(Archive &archive);

private:
   /// Indices of the source vertex for each edge
   RecordableVector<int> sourceVertexIndexCurrentEpoch_;

   /// Indices of the destination vertex for each edge
   RecordableVector<int> destVertexIndexCurrentEpoch_;

   /// The weight (scaling factor, strength, maximal amplitude) of each vertex for the current epoch.
   // vector<BGFLOAT> changes to RecordableVector for recording purpose
   RecordableVector<BGFLOAT> WCurrentEpoch_;
   // vector<BGFLOAT> WCurrentEpoch_;

   struct DistDestVertex {
      BGFLOAT dist;     ///< distance to the destination vertex
      int destVertex;   ///< index of the destination vertex

      bool operator<(const DistDestVertex &other) const
      {
         return (dist < other.dist);
      }
   };
};

CEREAL_REGISTER_TYPE(ConnStatic);

///  Cereal serialization method
template <class Archive> void ConnStatic::serialize(Archive &archive)
{
   archive(cereal::base_class<Connections>(this),
           cereal::make_nvp("sourceVertexIndexCurrentEpoch", sourceVertexIndexCurrentEpoch_),
           cereal::make_nvp("destVertexIndexCurrentEpoch", destVertexIndexCurrentEpoch_),
           cereal::make_nvp("WCurrentEpoch", WCurrentEpoch_));
}
