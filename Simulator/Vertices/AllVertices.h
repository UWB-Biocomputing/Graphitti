/**
 * @file AllVertices.h
 * 
 * @ingroup Simulation/Vertices
 *
 * @brief A container of the base class of all vertex data
 *
 * The class uses a data-centric structure, which utilizes a structure as the containers of
 * all vertices.
 *
 * The container holds vertex parameters of all vertices.
 * Each kind of vertex parameter is stored in a 1D array, of which length
 * is number of all vertices. Each array of a vertex parameter is pointed by a
 * corresponding member variable of the vertex parameter in the class.
 *
 * This structure was originally designed for the GPU implementation of the
 * simulator, and this refactored version of the simulator simply uses that design for
 * all other implementations as well. This is to simplify transitioning from
 * single-threaded to multi-threaded.
 */

#pragma once

using namespace std;

#include <log4cplus/loggingmacros.h>

#include "IAllVertices.h"
#include "BGTypes.h"

class AllVertices : public IAllVertices {
public:
   AllVertices();

   virtual ~AllVertices();

   ///  Setup the internal structure of the class.
   ///  Allocate memories to store all neurons' state.
   virtual void setupVertices();

   ///  Prints out all parameters of the neurons to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const;

   ///  The summation point for each vertex.
   ///  Summation points are places where the synapses connected to the vertex
   ///  apply (summed up) their PSRs (Post-Synaptic-Response).
   ///  On the next advance cycle, vertices add the values stored in their corresponding
   ///  summation points to their Vm and resets the summation points to zero
   BGFLOAT *summationMap_;

protected:
   ///  Total number of vertices.
   int size_;

   // Loggers used to print to using log4cplus logging macros
   log4cplus::Logger fileLogger_; // Logs to Output/Debug/logging.txt
   log4cplus::Logger vertexLogger_; // Logs to Output/Debug/neurons.txt
};

#if defined(USE_GPU)
struct AllVerticesDeviceProperties
{
        ///  The summation point for each vertex.
        ///  Summation points are places where the synapses connected to the vertex 
        ///  apply (summed up) their PSRs (Post-Synaptic-Response). 
        ///  On the next advance cycle, vertices add the values stored in their corresponding 
        ///  summation points to their Vm and resets the summation points to zero
        BGFLOAT *summationMap_;
};
#endif // defined(USE_GPU)
