/**
 *  @file AllEdges.h
 *
 *  @ingroup Simulator/Edges
 * 
 *  @brief A container of all edge data
 *
 *  The container holds edge parameters of all edges. 
 *  Each kind of edge parameter is stored in a 2D array. Each item in the first 
 *  dimention of the array corresponds with each vertex, and each item in the second
 *  dimension of the array corresponds with a edge parameter of each edge of the vertex. 
 *  Bacause each vertex owns different number of edges, the number of edges 
 *  for each vertex is stored in a 1D array, synapse_counts.
 *
 *  For CUDA implementation, we used another structure, AllEdgesDevice, where edge
 *  parameters are stored in 1D arrays instead of 2D arrays, so that device functions
 *  can access these data less latency. When copying a edge parameter, P[i][j],
 *  from host to device, it is stored in P[i * max_synapses_per_neuron + j] in 
 *  AllEdgesDevice structure.
 *
 *  The latest implementation uses the identical data struture between host and CUDA;
 *  that is, edge parameters are stored in a 1D array, so we don't need conversion 
 *  when copying data between host and device memory.
 */

#pragma once

#include <log4cplus/loggingmacros.h>

#include "Global.h"
#include "Core/Simulator.h"
#include "IAllEdges.h"

// cereal
#include <ThirdParty/cereal/types/vector.hpp>
#include <vector>

#ifdef _WIN32
typedef unsigned _int8 uint8_t;
#endif

class IAllVertices;

class AllEdges : public IAllEdges {
public:
   AllEdges();

   AllEdges(const int numVertices, const int maxSynapses);

   virtual ~AllEdges();

   ///  Setup the internal structure of the class (allocate memories and initialize them).
   virtual void setupEdges();

   /// Load member variables from configuration file.
   /// Registered to OperationManager as Operation::op::loadParameters
   virtual void loadParameters();

   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const;

   ///  Reset time varying state vars and recompute decay.
   ///
   ///  @param  iEdg     Index of the edge to set.
   ///  @param  deltaT   Inner simulation step duration
   virtual void resetEdge(const BGSIZE iEdg, const BGFLOAT deltaT);

   ///  Adds a Synapse to the model, connecting two Neurons.
   ///
   ///  @param  iEdg        Index of the edge to be added.
   ///  @param  type        The type of the Synapse to add.
   ///  @param  srcVertex   The Neuron that sends to this Synapse.
   ///  @param  destVertex  The Neuron that receives from the Synapse.
   ///  @param  sumPoint    Summation point address.
   ///  @param  deltaT      Inner simulation step duration
   virtual void
   addEdge(BGSIZE &iEdg, synapseType type, const int srcVertex, const int destVertex, BGFLOAT *sumPoint,
              const BGFLOAT deltaT);

   ///  Create a Synapse and connect it to the model.
   ///
   ///  @param  iEdg        Index of the edge to set.
   ///  @param  source      Coordinates of the source Neuron.
   ///  @param  dest        Coordinates of the destination Neuron.
   ///  @param  sumPoint    Summation point address.
   ///  @param  deltaT      Inner simulation step duration.
   ///  @param  type        Type of the Synapse to create.
   virtual void createEdge(const BGSIZE iEdg, int srcVertex, int destVertex, BGFLOAT *sumPoint, const BGFLOAT deltaT,
                              synapseType type) = 0;

   ///  Create a edge index map and returns it .
   ///
   /// @return the created EdgeIndexMap
   virtual EdgeIndexMap *createEdgeIndexMap();

   ///  Get the sign of the synapseType.
   ///
   ///  @param    type    synapseType I to I, I to E, E to I, or E to E
   ///  @return   1 or -1, or 0 if error
   int edgSign(const synapseType type);

   ///  Prints SynapsesProps data to console.
   virtual void printSynapsesProps() const;

   ///  Cereal serialization method
   ///  (Serializes edge weights, source vertices, and destination vertices)
   template<class Archive>
   void save(Archive &archive) const;

   ///  Cereal deserialization method
   ///  (Deserializes edge weights, source vertices, and destination vertices)
   template<class Archive>
   void load(Archive &archive);

protected:
   ///  Setup the internal structure of the class (allocate memories and initialize them).
   ///
   ///  @param  numVertices   Total number of vertices in the network.
   ///  @param  maxSynapses  Maximum number of edges per vertex.
   virtual void setupEdges(const int numVertices, const int maxSynapses);

   ///  Sets the data for Synapse to input's data.
   ///
   ///  @param  input  istream to read from.
   ///  @param  iEdg   Index of the edge to set.
   virtual void readEdge(istream &input, const BGSIZE iEdg);

   ///  Write the edge data to the stream.
   ///
   ///  @param  output  stream to print out to.
   ///  @param  iEdg    Index of the edge to print out.
   virtual void writeSynapse(ostream &output, const BGSIZE iEdg) const;

   ///  Returns an appropriate synapseType object for the given integer.
   ///
   ///  @param  typeOrdinal    Integer that correspond with a synapseType.
   ///  @return the SynapseType that corresponds with the given integer.
   synapseType synapseOrdinalToType(const int typeOrdinal);

   /// Loggers used to print to using log4cplus logging macros, prints to Results/Debug/logging.txt
   log4cplus::Logger fileLogger_;

#if !defined(USE_GPU)
public:
   ///  Advance all the Synapses in the simulation.
   ///  Update the state of all edges for a time step.
   ///
   ///  @param  vertices   The Neuron list to search from.
   ///  @param  edgeIndexMap   Pointer to EdgeIndexMap structure.
   virtual void advanceEdges(IAllVertices *vertices, EdgeIndexMap *edgeIndexMap);

   ///  Remove a edge from the network.
   ///
   ///  @param  neuronIndex   Index of a vertex to remove from.
   ///  @param  iEdg           Index of a edge to remove.
   virtual void eraseEdge(const int neuronIndex, const BGSIZE iEdg);

#endif // !defined(USE_GPU)
public:
   /// The factor to adjust overlapping area to edge weight.
   static constexpr BGFLOAT SYNAPSE_STRENGTH_ADJUSTMENT = 1.0e-8;

   ///  The location of the edge.
   int *sourceNeuronIndex_;

   ///  The coordinates of the summation point.
   int *destNeuronIndex_;

   ///   The weight (scaling factor, strength, maximal amplitude) of the edge.
   BGFLOAT *W_;

   ///  This edge's summation point's address.
   BGFLOAT **summationPoint_;

   ///   Synapse type
   synapseType *type_;

   ///  The post-synaptic response is the result of whatever computation
   ///  is going on in the edge.
   BGFLOAT *psr_;

   ///  The boolean value indicating the entry in the array is in use.
   bool *inUse_;

   ///  The number of (incoming) edges for each vertex.
   ///  Note: Likely under a different name in GpuSim_struct, see edge_count. -Aaron
   BGSIZE *synapseCounts_;

   ///  The total number of active edges.
   BGSIZE totalEdgeCount_;

   ///  The maximum number of edges for each vertex.
   BGSIZE maxEdgesPerVertex_;

   ///  The number of vertices
   ///  Aaron: Is this even supposed to be here?!
   ///  Usage: Used by destructor
   int countVertices_;
};

#if defined(USE_GPU)
struct AllSynapsesDeviceProperties
{
        ///  The location of the edge.
        int *sourceNeuronIndex_;

        ///  The coordinates of the summation point.
        int *destNeuronIndex_;

        ///   The weight (scaling factor, strength, maximal amplitude) of the edge.
         BGFLOAT *W_;

        ///  Synapse type
        synapseType *type_;

        ///  The post-synaptic response is the result of whatever computation
        ///  is going on in the edge.
        BGFLOAT *psr_;

        ///  The boolean value indicating the entry in the array is in use.
        bool *inUse_;

        ///  The number of edges for each vertex.
        ///  Note: Likely under a different name in GpuSim_struct, see edge_count. -Aaron
        BGSIZE *synapseCounts_;

        ///  The total number of active edges.
        BGSIZE totalEdgeCount_;

        ///  The maximum number of edges for each vertex.
        BGSIZE maxEdgesPerVertex_;

        ///  The number of vertices
        ///  Aaron: Is this even supposed to be here?!
        ///  Usage: Used by destructor
        int countVertices_;
};
#endif // defined(USE_GPU)

///  Cereal serialization method
///  (Serializes edge weights, source vertices, and destination vertices)
template<class Archive>
void AllEdges::save(Archive &archive) const {
   // uses vector to save edge weights, source vertices, and destination vertices
   vector<BGFLOAT> WVector;
   vector<int> sourceNeuronLayoutIndexVector;
   vector<int> destNeuronLayoutIndexVector;

   for (int i = 0; i < maxEdgesPerVertex_ * countVertices_; i++) {
      WVector.push_back(W_[i]);
      sourceNeuronLayoutIndexVector.push_back(sourceNeuronIndex_[i]);
      destNeuronLayoutIndexVector.push_back(destNeuronIndex_[i]);
   }

   // serialization
   archive(WVector, sourceNeuronLayoutIndexVector, destNeuronLayoutIndexVector);
}

///  Cereal deserialization method
///  (Deserializes edge weights, source vertices, and destination vertices)
template<class Archive>
void AllEdges::load(Archive &archive) {
   // uses vectors to load edge weights, source vertices, and destination vertices
   vector<BGFLOAT> WVector;
   vector<int> sourceNeuronLayoutIndexVector;
   vector<int> destNeuronLayoutIndexVector;

   // deserializing data to these vectors
   archive(WVector, sourceNeuronLayoutIndexVector, destNeuronLayoutIndexVector);

   // check to see if serialized data sizes matches object sizes
   if (WVector.size() != maxEdgesPerVertex_ * countVertices_) {
      cerr
            << "Failed deserializing edge weights, source vertices, and/or destination vertices. Please verify maxEdgesPerVertex and count_neurons data members in AllEdges class."
            << endl;
      throw cereal::Exception("Deserialization Error");
   }

   // assigns serialized data to objects
   for (int i = 0; i < maxEdgesPerVertex_ * countVertices_; i++) {
      W_[i] = WVector[i];
      sourceNeuronIndex_[i] = sourceNeuronLayoutIndexVector[i];
      destNeuronIndex_[i] = destNeuronLayoutIndexVector[i];
   }
}