/**
 * @file ConnStatic.h
 *
 * @ingroup Simulation/Connections
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
 *
 * Some models in this simulator is a rewrite of CSIM (2006) and other
 * work (Stiber and Kawasaki (2007?))
 */

#pragma once

#include "Global.h"
#include "Connections.h"
#include "Simulator.h"
#include <vector>
#include <iostream>

using namespace std;

class ConnStatic : public Connections {
public:
   ConnStatic();

   virtual ~ConnStatic();

   static Connections *Create() { return new ConnStatic(); }

   ///  Setup the internal structure of the class (allocate memories and initialize them).
   ///  Initialize the small world network characterized by parameters:
   ///  number of maximum connections per neurons, connection radius threshold, and
   ///  small-world rewiring probability.
   ///
   ///  @param  layout    Layout information of the neunal network.
   ///  @param  neurons   The Neuron list to search from.
   ///  @param  synapses  The Synapse list to search from.
   virtual void setupConnections(Layout *layout, IAllNeurons *neurons, IAllSynapses *synapses);

   /// Load member variables from configuration file.
   /// Registered to OperationManager as Operations::op::loadParameters
   virtual void loadParameters();

   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const;

private:
   /// number of maximum connections per neurons
   int connsPerNeuron_;

   /// Connection radius threshold
   BGFLOAT threshConnsRadius_;

   /// Small-world rewiring probability
   BGFLOAT rewiringProbability_;

   /// Min/max values of excitatory neuron's synapse weight
   BGFLOAT excWeight_[2];

   /// Min/max values of inhibitory neuron's synapse weight
   BGFLOAT inhWeight_[2];

   struct DistDestNeuron {
      BGFLOAT dist;     ///< destance to the destination neuron
      int destNeuron;  ///< index of the destination neuron

      bool operator<(const DistDestNeuron &other) const {
         return (dist < other.dist);
      }
   };
};
