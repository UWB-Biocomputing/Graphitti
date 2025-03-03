/**
 * @file All911Vertices.h
 * 
 * @ingroup Simulator/Vertices/NG911
 *
 * @brief Specialization of the AllVertices class for the NG911 network
 *
 * The data-centric class uses a container structure for all vertices.
 *
 * The container holds vertex parameters for all vertices.
 * Each kind of vertex parameter is stored in a 1D vector, of which length
 * is number of all vertices.
 * 
 * The NG911 is modeled as a multiple queueing systems. Each vertex contains
 * a queue of events (calls) that are processed in a first-in first-out FIFO
 * fashion, and the processing of the events depends on the type of vertex.
 *
 * Currently, we are modeling 5 vertexTypes: CALR, PSAP, EMS, FIRE, and LAW.
 * EMS, FIRE, and LAW represent types of Emergency Responders; while CALR and
 * PSAP represent a Caller Region and a Public Safety Answering Point (PSAP),
 * respectively.
 * 
 * In general, Calls that originate at a Caller Region pass first through a
 * queue at a PSAP where a call taker services the them, and then dispatches
 * an appropriate responder. Once the call is assigned to a dispatcher, it goes
 * through another queue at the Responder vertex waiting for an availale
 * responder unit. The call taker will dispatch a responder unit from the
 * responder closest to the emergency event.
 * 
 * There are a few stochastic processes involved in the NG911 models of which
 * the call arrival rate is one of the most complex. We want to allow the use
 * of different arrival rate models and also be able to recreate pass events
 * from real world 911 data. For this reason, we decided to use an external
 * input file to provide the stream of calls comming into the system. This
 * allows us to support real world data, and also experiment with different call
 * arrival models without modifying the code.
 * 
 * We support the aforementioned input file throught an InputManager object
 * that loads the events from an XML file. The vertices load the call events to
 * be process during an epoch, at the beginning of that epoch. We need to
 * inform the InputManager object of the information associated with each call
 * by registering those properties in the `setupVertices` method. After that,
 * once all the vertices are created the InputManager object also loads the
 * external input file.
 * 
 * We also implement call abandonment, which is when a caller gets tired of
 * waiting in the queue and decides to abandon it. This is supported through the
 * `patience` call property, which represent the amount of time a caller is
 * willing to waiting before abandoing the queue. If their time in the queue exceeds
 * this `patience` time, the call is considered abandoned.
 * 
 * Redialing is implemented using a redial probability (`redialP`) in the
 * configuration file. We assume that when a caller decides to redial, he does
 * so immediately and will try until getting through.
 * 
 * The naming of some variables come from either operations research in call
 * centers, or queueing theory. For instance, `trunks` represent phone lines
 * available for incoming calls while `servers` represent the resources that
 * are tied up while processing a call. Servers in a PSAP are call taker, while
 * in a responder node they are responder units. 
 * 
 * Finally, we size the capacity of the waiting queue to the number of trunks
 * and keep track of the number of busy agents at every time step to estimate
 * when the queue is full.
 */
#pragma once

#include "AllVertices.h"
#include "CircularBuffer.h"
#include "Global.h"
#include "GraphManager.h"
#include "InputEvent.h"
#include "InputManager.h"

// Forward declaration to avoid circular reference
class All911Edges;

// Class to hold all data necessary for all the Vertices.
class All911Vertices : public AllVertices {
public:
   // Xml911Recorder needs to access some of this class private members
   friend class Xml911Recorder;

   All911Vertices() = default;

   virtual ~All911Vertices() = default;

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
   virtual void createAllVertices(Layout &layout);

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
   virtual string toString(int index) const;

   /// Loads all inputs scheduled to occur in the upcoming epoch.
   /// These are inputs occurring in between curStep (inclusive) and
   /// endStep (exclusive)
   virtual void loadEpochInputs(uint64_t currentStep, uint64_t endStep) override;

   /// unused virtual function placeholder
   virtual void registerHistoryVariables() override
   {
   }

   /// Accessor for the waiting queue of a vertex
   ///
   /// @param vIdx   The index of the vertex
   /// @return    The waiting queue for the given vertex
   CircularBuffer<Call> &getQueue(int vIdx);

   /// Accessor for the droppedCalls counter of a vertex
   ///
   /// @param vIdx   The index of the vertex
   /// @return    A reference to the droppedCalls counter of the vertex
   int &droppedCalls(int vIdx);

   /// Accessor for the receivedCalls counter of a vertex
   ///
   /// @param vIdx   The index of the vertex
   /// @return    A reference to the receivedCalls counter of the vertex
   int &receivedCalls(int vIdx);

   /// Accessor for the number of busy servers in a given vertex
   ///
   /// @param vIdx   The index of the vertex
   /// @return    The number of busy servers in the given vertex
   int busyServers(int vIdx) const;

private:
   /// The starting time for every call
   vector<vector<uint64_t>> beginTimeHistory_;
   /// The answer time for every call
   vector<vector<uint64_t>> answerTimeHistory_;
   /// The end time for every call
   vector<vector<uint64_t>> endTimeHistory_;
   /// True if the call was abandoned
   vector<vector<unsigned char>> wasAbandonedHistory_;
   /// The length of the waiting queue at every time-step
   vector<vector<int>> queueLengthHistory_;
   /// The portion of servers that are busy at every time-step
   vector<vector<double>> utilizationHistory_;

   /// These are the queues where calls will wait to be served
   vector<CircularBuffer<Call>> vertexQueues_;

   /// The number of calls that have been dropped (got a busy signal)
   vector<int> droppedCalls_;

   /// The number of received calls
   vector<int> receivedCalls_;

   /// Number of servers currently serving calls
   vector<int> busyServers_;

   /// Number of servers. In a PSAP these are the call takers, in Responder nodes
   /// they are responder units
   vector<int> numServers_;

   /// Number of phone lines available. Only valid for PSAPs and Responders
   vector<int> numTrunks_;

   /// The probability that a caller will redial after receiving the busy signal
   BGFLOAT redialP_;

   /// The average driving speed of a response unit in mph
   BGFLOAT avgDrivingSpeed_;

   /// Holds the calls being served by each server
   vector<vector<Call>> servingCall_;

   /// The time that the call being served was answered by the server
   vector<vector<uint64_t>> answerTime_;

   /// The countdown until the server is available to take another call
   vector<vector<int>> serverCountdown_;

   /// The InputManager holds all the Input Events for the simulation
   InputManager<Call> inputManager_;

   ///  Advance a CALR vertex. Send calls to the appropriate PSAP
   ///
   ///  @param  vertexIdx     Index of the CALR vertex
   ///  @param  edgeIndexMap  Reference to the EdgeIndexMap.
   ///  @param  allEdges      Reference to an instance of All911Edges
   void advanceCALR(BGSIZE vertexIdx, All911Edges &edges911, const EdgeIndexMap &edgeIndexMap);

   ///  Advance a PSAP vertex. Controls the redirection and handling of calls
   ///
   ///  @param  vertexIdx     Index of the PSAP vertex
   ///  @param  edgeIndexMap  Reference to the EdgeIndexMap.
   ///  @param  allEdges      Reference to an instance of All911Edges
   void advancePSAP(BGSIZE vertexIdx, All911Edges &edges911, const EdgeIndexMap &edgeIndexMap);

   ///  Advance a RESP vertex. Receives call from PSAP and responds to the emergency events
   ///
   ///  @param  vertexIdx     Index of the RESP vertex
   ///  @param  edgeIndexMap  Reference to the EdgeIndexMap.
   ///  @param  allEdges      Reference to an instance of All911Edges
   void advanceRESP(BGSIZE vertexIdx, All911Edges &edges911, const EdgeIndexMap &edgeIndexMap);

#if defined(USE_GPU)
   // GPU functionality for 911 simulation is unimplemented.
   // These signatures are required to make the class non-abstract
public:
   virtual void allocNeuronDeviceStruct(void **allVerticesDevice) {};
   virtual void deleteNeuronDeviceStruct(void *allVerticesDevice) {};
   virtual void copyToDevice(void *allVerticesDevice) {};
   virtual void copyFromDevice(void *allVerticesDevice) {};
   virtual void advanceVertices(AllEdges &edges, void *allVerticesDevice, void *allEdgesDevice,
                                float randNoise[], EdgeIndexMapDevice *edgeIndexMapDevice) {};
   virtual void setAdvanceVerticesDeviceParams(AllEdges &edges) {};
#else   // !defined(USE_GPU)
public:
   ///  Update internal state of the indexed Vertex (called by every simulation step).
   ///  Notify outgoing edges if vertex has fired.
   ///
   ///  @param  edges         The Edge list to search from.
   ///  @param  edgeIndexMap  Reference to the EdgeIndexMap.
   virtual void advanceVertices(AllEdges &edges, const EdgeIndexMap &edgeIndexMap) override;

protected:

#endif   // defined(USE_GPU)
};