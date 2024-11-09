/**
 * @file Layout911.h
 * 
 * @ingroup Simulator/Layouts/NG911
 *
 * @brief Specialization of the Layout class for the NG911 network
 *
 * The Layout911 class maintains vertices locations (x, y coordinates), distance
 * of every pair of vertices, a map of the vertices types, and starter vertices
 * map.
 *
 * All the information about the vertices is loaded from a GraphML input file
 * using the GraphManager class. Layout911 registers the vertex properties
 * with the GraphManager, a singleton, which then loads the graph defined in
 * the GraphML input file. Subsequently, Layout911 uses the GraphManager to
 * create a layout of the NG911 nodes used for the simulation.
 * 
 * The GraphManager is only used as a middle-man to facilitate the loading
 * of the initial graph, defined in the GraphML file. After this initialization
 * step, the layout contained within Layout911 is the one used throughout the
 * Simulation.
 * 
 * Currently, we are using 5 vertexTypes in the NG911 models: CALR, PSAP, EMS,
 * FIRE, and LAW. EMS, FIRE, and LAW represent types of Emergency Responders;
 * while CALR and PSAP represent a Caller Region and a Public Safety Answering
 * Point (PSAP), respectively.
 * 
 * Layout911 is in charge of loading and managing the vertices layout while the
 * All911Vertices class holds the internal behaviour of all vertices.
 */

#pragma once

#include "Layout.h"

using namespace std;

class Layout911 : public Layout {
public:
   Layout911() = default;

   virtual ~Layout911() = default;

   /// Creates an instance of the class.
   ///
   /// @return Reference to the instance of the class.
   static Layout *Create()
   {
      return new Layout911();
   }

   /// Register vertex properties with the GraphManager
   virtual void registerGraphProperties() override;

   /// Loads Layout911 member variables.
   /// Registered to OperationManager as Operation::loadParameters
   virtual void loadParameters() override;

   /// Setup the internal structure of the class.
   /// Allocate memories to store all layout state.
   virtual void setup() override;

   /// Prints out all parameters to logging file.
   /// Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const override;

   /// Creates a vertex type map.
   ///
   /// @param  numVertices number of the vertices to have in the type map.
   virtual void generateVertexTypeMap() override;

   /// Returns the type of synapse at the given coordinates
   /// @param    srcVertex  integer that points to a Neuron in the type map as a source.
   /// @param    destVertex integer that points to a Neuron in the type map as a destination.
   /// @return type of the synapse.
   virtual edgeType edgType(int srcVertex, int destVertex) override;

   /// Calculates the distance between the given vertex and the (x, y) coordinates of a point
   /// @param vertexId  The index of the vertex to calculate the distance from
   /// @param x         The x location of a point
   /// @param y         The y location of a point
   /// @return The distance between the given vertex and the (x, y) coordinates of a point
   double getDistance(int vertexId, double x, double y);
};
