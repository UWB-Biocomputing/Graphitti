/**
 *      @file AllNeurons.h
 *
 *      @brief A container of the base class of all neuron data
 */

/**
 ** @class AllNeurons AllNeurons.h "AllNeurons.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** A container of the base class of all neuron data.
 **
 ** The class uses a data-centric structure, which utilizes a structure as the containers of
 ** all neuron.
 **
 ** The container holds neuron parameters of all neurons.
 ** Each kind of neuron parameter is stored in a 1D array, of which length
 ** is number of all neurons. Each array of a neuron parameter is pointed by a
 ** corresponding member variable of the neuron parameter in the class.
 **
 ** This structure was originally designed for the GPU implementation of the
 ** simulator, and this refactored version of the simulator simply uses that design for
 ** all other implementations as well. This is to simplify transitioning from
 ** single-threaded to multi-threaded.
 **
 ** \latexonly  \subsubsection*{Credits} \endlatexonly
 ** \htmlonly   <h3>Credits</h3> \endhtmlonly
 **
 ** Some models in this simulator is a rewrite of CSIM (2006) and other
 ** work (Stiber and Kawasaki (2007?))
 **
 **/

#pragma once

using namespace std;

#include <log4cplus/loggingmacros.h>

#include "IAllNeurons.h"
#include "BGTypes.h"

class AllNeurons : public IAllNeurons {
public:
   AllNeurons();

   virtual ~AllNeurons();

   /**
    *  Setup the internal structure of the class.
    *  Allocate memories to store all neurons' state.
    *
    */
   virtual void setupNeurons();

   /**
    *  Prints out all parameters of the neurons to logging file.
    *  Registered to OperationManager as Operation::printParameters
    */
   virtual void printParameters() const;

   /**
    *  The summation point for each neuron.
    *  Summation points are places where the synapses connected to the neuron
    *  apply (summed up) their PSRs (Post-Synaptic-Response).
    *  On the next advance cycle, neurons add the values stored in their corresponding
    *  summation points to their Vm and resets the summation points to zero
    */
   BGFLOAT *summationMap_;

protected:
   /**
    *  Total number of neurons.
    */
   int size_;

   // Loggers used to print to using log4cplus logging macros
   log4cplus::Logger fileLogger_; // Logs to Output/Debug/logging.txt
   log4cplus::Logger neuronLogger_; // Logs to Output/Debug/neurons.txt
};

#if defined(USE_GPU)
struct AllNeuronsDeviceProperties
{
        /** 
         *  The summation point for each neuron.
         *  Summation points are places where the synapses connected to the neuron 
         *  apply (summed up) their PSRs (Post-Synaptic-Response). 
         *  On the next advance cycle, neurons add the values stored in their corresponding 
         *  summation points to their Vm and resets the summation points to zero
         */
        BGFLOAT *summationMap_;
};
#endif // defined(USE_GPU)
