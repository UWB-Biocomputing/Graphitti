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

   ///  Prints out all parameters of the vertices to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const;

#if defined(USE_GPU)
   public:

       ///  Update the state of all vertices for a time step
       ///  Notify outgoing edges if vertex has fired.
       ///
       ///  @param  edges               Reference to the allEdges struct on host memory.
       ///  @param  allVerticesDevice       GPU address of the allNeurons struct on device memory.
       ///  @param  allEdgesDevice      GPU address of the allEdges struct on device memory.
       ///  @param  randNoise              Reference to the random noise array.
       ///  @param  edgeIndexMapDevice  GPU address of the EdgeIndexMap on device memory.
       virtual void advanceVertices(IAllEdges &edges, void* allVerticesDevice, void* allEdgesDevice, float* randNoise, EdgeIndexMap* edgeIndexMapDevice);

#else  // !defined(USE_GPU)
protected:

#endif // defined(USE_GPU)
};

