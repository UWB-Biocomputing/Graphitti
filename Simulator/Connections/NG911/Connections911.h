/**
 * @file Connections911.h
 *
 * @ingroup Simulator/Connections/NG911
 * 
 * @brief The model of the static network
 *
 */

#pragma once

#include "Global.h"
#include "Connections.h"
#include "Simulator.h"
#include <vector>

using namespace std;

class Connections911 : public Connections {
public:
   Connections911();

   virtual ~Connections911();

   ///  Creates an instance of the class.
   ///
   ///  @return Reference to the instance of the class.
   static Connections *Create() { return new Connections911(); }

   ///  Setup the internal structure of the class (allocate memories and initialize them).
   ///  Initialize the network characterized by parameters:
   ///  number of maximum connections per vertex, connection radius threshold
   ///
   ///  @param  layout    Layout information of the network.
   ///  @param  vertices   The Vertex list to search from.
   ///  @param  edges  The edge list to search from.
   virtual void setupConnections(Layout *layout, IAllVertices *vertices, IAllEdges *edges);

   /// Load member variables from configuration file.
   /// Registered to OperationManager as Operations::op::loadParameters
   virtual void loadParameters();

   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const;

private:
   /// number of maximum connections per vertex
   int connsPerVertex_;

   /// Connection radius threshold
   BGFLOAT threshConnsRadius_;

   struct DistDestVertex {
      BGFLOAT dist;     ///< destance to the destination vertex
      int destVertex;  ///< index of the destination vertex

      bool operator<(const DistDestVertex &other) const {
         return (dist < other.dist);
      }
   };
};
