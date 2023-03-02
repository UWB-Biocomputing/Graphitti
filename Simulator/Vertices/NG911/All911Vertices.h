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

#include "AllVertices.h"
#include "CircularBuffer.h"
#include "Global.h"
#include "InputEvent.h"
#include "InputManager.h"

// Class to hold all data necessary for all the Vertices.
class All911Vertices : public AllVertices {
public:
   All911Vertices();

   virtual ~All911Vertices();

   ///  Creates an instance of the class.
   ///
   ///  @return Reference to the instance of the class.
   static AllVertices *Create()
   {
      return new All911Vertices();
   }

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
   virtual void printParameters() const override;

   ///  Outputs state of the vertex chosen as a string.
   ///
   ///  @param  index   index of the vertex (in vertices) to output info from.
   ///  @return the complete state of the vertex.
   virtual string toString(const int index) const;

   /// Loads all inputs scheduled to occur in the upcoming epoch.
   /// These are inputs occurring in between curStep (inclusive) and
   /// endStep (exclusive)
   virtual void loadEpochInputs(uint64_t curStep, uint64_t endStep) override;

   /// These are the queues where calls will wait to be served
   vector<CircularBuffer<Call>> vertexQueues_;

   /// The number of calls that have been dropped (got a busy signal)
   vector<int> droppedCalls_;

   /// The number of received calls
   vector<int> receivedCalls_;

private:
   /// Number of agents. In a PSAP these are the call takers, in Responder nodes
   /// they are responder units
   vector<int> numAgents_;

   /// Number of phone lines available. Only valid for PSAPs and Responders
   vector<int> numTrunks_;

   /// Holds the calls being served by each agent
   vector<vector<Call>> servingCall_;

   /// The time that the call being served was answered by the agent
   vector<vector<uint64_t>> answerTime_;

   /// The countdown until the agent is available to take another call
   vector<vector<int>> agentCountdown_;

   /// The InputManager holds all the Input Events for the simulation
   InputManager<Call> inputManager_;

   log4cplus::Logger fileLogger_;      // For logging into a file
   log4cplus::Logger consoleLogger_;   // For logging to console

   // TODO: The variables below are from previous version. I need to review what
   //       they are for and if they are being used anywhere.

   /// number of callers
   int *callNum_;

   /// Min/max values of CallNum.
   int callNumRange_[2];

   /// Number of dispatchers per PSAP calculated (with randomness) based on population
   int *dispNum_;

   /// Scaling factor for number of dispatchers in a PSAP
   BGFLOAT dispNumScale_;

   /// Number of responders per Responder node calculated (with randomness) based on population
   int *respNum_;

   /// Scaling factor for number of responders in a Responder node
   BGFLOAT respNumScale_;

#if defined(USE_GPU)
   // GPU functionality for 911 simulation is unimplemented.
   // These signatures are required to make the class non-abstract
public:
   virtual void allocNeuronDeviceStruct(void **allVerticesDevice) {};
   virtual void deleteNeuronDeviceStruct(void *allVerticesDevice) {};
   virtual void copyToDevice(void *allVerticesDevice) {};
   virtual void copyFromDevice(void *allVerticesDevice) {};
   virtual void advanceVertices(AllEdges &edges, void *allVerticesDevice, void *allEdgesDevice,
                                float *randNoise, EdgeIndexMap *edgeIndexMapDevice) {};
   virtual void setAdvanceVerticesDeviceParams(AllEdges &edges) {};
#else   // !defined(USE_GPU)
public:
   ///  Update internal state of the indexed Vertex (called by every simulation step).
   ///  Notify outgoing edges if vertex has fired.
   ///
   ///  @param  edges         The Edge list to search from.
   ///  @param  edgeIndexMap  Reference to the EdgeIndexMap.
   virtual void advanceVertices(AllEdges &edges, const EdgeIndexMap *edgeIndexMap) override;

protected:
   void advanceVertex(const int index);

#endif   // defined(USE_GPU)

};
