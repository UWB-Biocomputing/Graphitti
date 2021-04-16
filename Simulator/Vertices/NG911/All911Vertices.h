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

   ///  Creates an instance of the class.
   ///
   ///  @return Reference to the instance of the class.
   static IAllVertices *Create() { return new All911Vertices(); }

   ///  Setup the internal structure of the class.
   ///  Allocate memories to store all vertices' states.
   virtual void allocateMemory();

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

   ///  Outputs state of the vertex chosen as a string.
   ///
   ///  @param  index   index of the vertex (in vertices) to output info from.
   ///  @return the complete state of the vertex.
   virtual string toString(const int index) const;


protected: 
   ///  Creates a single vertex and generates data for it.
   ///
   ///  @param  index   Index of the vertex to create.
   ///  @param  layout  Layout information of the network.
   void createVertex(int index, Layout *layout);

private: 

   /// number of callers
   int *CallNum_; 

   /// Min/max values of CallNum.
   int CallNumRange_[2];

#if defined(USE_GPU)

#else  // !defined(USE_GPU)
public:
 
   ///  Update internal state of the indexed Vertex (called by every simulation step).
   ///  Notify outgoing edges if vertex has fired.
   ///
   ///  @param  edges         The Edge list to search from.
   ///  @param  edgeIndexMap  Reference to the EdgeIndexMap.
   virtual void advanceVertices(AllEdges &edges, const EdgeIndexMap *edgeIndexMap);

protected:

#endif // defined(USE_GPU)
};

