/**
 * @file All911Vertices.h
 * 
 * @ingroup Simulator/Vertices/NG911
 *
 * @brief A container of all 911 vertex data
 *
 * A container of all vertex data.
 *
 * The data-centric class uses a container structure for all vertices.
 *
 * The container holds vertex parameters for all vertices.
 * Each kind of vertex parameter is stored in a 1D array, of which length
 * is number of all vertices. Each array of a vertex parameter is pointed to by a
 * corresponding member variable of the vertex parameter in the class.
 *
 */
#pragma once

#include "Global.h"
#include "AllVertices.h"

// Class to hold all data necessary for all the Vertices.
class All911Vertices : public AllVertices {
public:

   All911Vertices();

   virtual ~All911Vertices();

   ///  Setup the internal structure of the class.
   ///  Allocate memories to store all vertices' states.
   virtual void setupVertices();

   ///  Creates all the Vertices and assigns initial data for them.
   ///
   ///  @param  layout      Layout information of the network.
   virtual void createAllVertices(Layout *layout);

   ///  Load member variables from configuration file.
   ///  Registered to OperationManager as Operation::loadParameters
   virtual void loadParameters();

   ///  Prints out all parameters of the vertices to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const;

protected: 
   ///  Creates a single vertex and generates data for it.
   ///
   ///  @param  index   Index of the vertex to create.
   ///  @param  layout  Layout information of the network.
   void createVertex(int index, Layout *layout);

#if defined(USE_GPU)

#else  // !defined(USE_GPU)
protected:

#endif // defined(USE_GPU)
};

