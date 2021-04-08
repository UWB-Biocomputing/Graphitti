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

   static Layout *Create() { return new Layout911(); }

   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const;

   ///  Creates a vertex type map.
   ///
   ///  @param  numVertices number of the vertices to have in the type map.
   virtual void generateVertexTypeMap(int numVertices);

   ///  Populates the starter map.
   ///
   ///  @param  numVertices number of vertices to have in the map.
   virtual void initStarterMap(const int numVertices);
};

