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
   virtual void generateVertexTypeMap(int numVertices) override;

   /// Populates the starter map.
   ///
   /// @param  numVertices number of vertices to have in the map.
   virtual void initStarterMap(const int numVertices) override;

   /// Get the zone of the vertex
   /// Only built for 10x10 grid
   /// See: https://docs.google.com/spreadsheets/d/1DqP8sjkfJ_pkxtETzuEdoVZbWOGu633EMQAeShe5k68/edit?usp=sharing
   /// @param  index    the index of the vertex
   int zone(int index);

   /// Returns the type of synapse at the given coordinates
   /// @param    srcVertex  integer that points to a Neuron in the type map as a source.
   /// @param    destVertex integer that points to a Neuron in the type map as a destination.
   /// @return type of the synapse.
   virtual edgeType edgType(const int srcVertex, const int destVertex) override;
};
