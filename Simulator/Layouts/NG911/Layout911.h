/**
 * @file Layout911.h
 * 
 * @ingroup Simulator/Layouts/NG911
 *
 * @brief The Layout class defines the layout of vertices in networks
 *
 * The Layout911 class maintains vertices locations (x, y coordinates), 
 * distance of every couple vertices,
 * vertices type map (distribution of vertex types), and starter vertices map
 *
 * The Layout911 class reads all layout information from parameter description file.
 */

#pragma once

#include "Layout.h"

using namespace std;

class Layout911 : public Layout {
public:
   Layout911();

   virtual ~Layout911();

   ///  Creates an instance of the class.
   ///
   ///  @return Reference to the instance of the class.
   static Layout *Create() { return new Layout911(); }

   /// Load member variables from configuration file. Registered to OperationManager as Operation::loadParameters
   virtual void loadParameters() override;

   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const override;

   ///  Creates a vertex type map.
   ///
   ///  @param  numVertices number of the vertices to have in the type map.
   virtual void generateVertexTypeMap(int numVertices) override;

   ///  Populates the starter map.
   ///
   ///  @param  numVertices number of vertices to have in the map.
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

