/**
 *  @file AllDSSynapses.h
 * 
 *  @ingroup Simulator/Edges
 *
 *  @brief A container of all DS synapse data
 *
 *  The container holds synapse parameters of all synapses. 
 *  Each kind of synapse parameter is stored in a 2D array. Each item in the first 
 *  dimention of the array corresponds with each neuron, and each item in the second
 *  dimension of the array corresponds with a synapse parameter of each synapse of the neuron. 
 *  Bacause each neuron owns different number of synapses, the number of synapses 
 *  for each neuron is stored in a 1D array, edge_counts.
 *
 *  For CUDA implementation, we used another structure, AllDSSynapsesDevice, where synapse
 *  parameters are stored in 1D arrays instead of 2D arrays, so that device functions
 *  can access these data less latency. When copying a synapse parameter, P[i][j],
 *  from host to device, it is stored in P[i * max_edges_per_vertex + j] in 
 *  AllDSSynapsesDevice structure.
 *
 *  The latest implementation uses the identical data struture between host and CUDA;
 *  that is, synapse parameters are stored in a 1D array, so we don't need conversion
 *  when copying data between host and device memory.
 *
 * Phenomenological model of frequency-dependent synapses exibit dynamics that include
 * activity-dependent facilitation and depression (Tsodyks and Markram 1997, Tsodyks et al. 1998).
 * The model has two state variables: \f$r\f$ (the fraction of available synaptic efficacy), and
 * \f$u\f$ (the running value of utilization of synaptic efficacy).
 *
 * \f[
 *  r_{n+1} = r_n \cdot (1-u_{n+1}) \cdot \mathrm{e}^{-\frac{\Delta t}{\tau_{rec}}} +
 *  1 - \mathrm{e}^{-\frac{\Delta t}{\tau_{rec}}}
 * \f]
 *
 * \f[
 * \f]
 *
 * where \f$\Delta t\f$ is the time interval between nth and (n + 1)th AP,
 * the two time constants \f$\tau_{rec}\f$ and \f$\tau_{facil}\f$ govern recovery from depression,
 * and facilitation after a spike, and \f$U\f$ is utilization of synaptic efficacy (Markram et al. 1998).
 *
 * The synaptic response that is generated by any AP in a train is therefore given by:
 * \f[
 *  EPSP_n = A \cdot r_n \cdot u_n
 * \f]
 */
#pragma once

#include "AllSpikingSynapses.h"

struct AllDSSynapsesDeviceProperties;

class AllDSSynapses : public AllSpikingSynapses {
public:
   AllDSSynapses();

   AllDSSynapses(const int numVertices, const int maxEdges);

   virtual ~AllDSSynapses();

   static AllEdges *Create() { return new AllDSSynapses(); }

   ///  Setup the internal structure of the class (allocate memories and initialize them).
   virtual void allocateMemory();

   ///  Reset time varying state vars and recompute decay.
   ///
   ///  @param  iEdg     Index of the synapse to set.
   ///  @param  deltaT   Inner simulation step duration
   virtual void resetEdge(const BGSIZE iEdg, const BGFLOAT deltaT);

   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const;

   ///  Create a Synapse and connect it to the model.
   ///
   ///  @param  iEdg        Index of the synapse to set.
   ///  @param  srcVertex     Coordinates of the source Neuron.
   ///  @param  destVertex        Coordinates of the destination Neuron.
   ///  @param  sumPoint   Summation point address.
   ///  @param  deltaT      Inner simulation step duration.
   ///  @param  type        Type of the Synapse to create.
   virtual void createEdge(const BGSIZE iEdg, int srcVertex, int destVertex, BGFLOAT *sumPoint, const BGFLOAT deltaT,
                              edgeType type);

   ///  Prints SynapsesProps data to console.
   virtual void printSynapsesProps() const;

protected:
   ///  Setup the internal structure of the class (allocate memories and initialize them).
   ///
   ///  @param  numVertices   Total number of vertices in the network.
   ///  @param  maxEdges  Maximum number of synapses per neuron.
   virtual void allocateMemory(const int numVertices, const int maxEdges);

   ///  Sets the data for Synapse to input's data.
   ///
   ///  @param  input  istream to read from.
   ///  @param  iEdg   Index of the synapse to set.
   virtual void readEdge(istream &input, const BGSIZE iEdg);

   ///  Write the synapse data to the stream.
   ///
   ///  @param  output  stream to print out to.
   ///  @param  iEdg    Index of the synapse to print out.
   virtual void writeEdge(ostream &output, const BGSIZE iEdg) const;

#if defined(USE_GPU)
   public:
       ///  Allocate GPU memories to store all synapses' states,
       ///  and copy them from host to GPU memory.
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       virtual void allocEdgeDeviceStruct( void** allEdgesDevice);

       ///  Allocate GPU memories to store all synapses' states,
       ///  and copy them from host to GPU memory.
       ///
       ///  @param  allEdgesDevice     GPU address of the allEdges struct on device memory.
       ///  @param  numVertices            Number of vertices.
       ///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
       virtual void allocEdgeDeviceStruct( void** allEdgesDevice, int numVertices, int maxEdgesPerVertex);

       ///  Delete GPU memories.
       ///
       ///  @param  allEdgesDevice     GPU address of the allEdges struct on device memory.
       virtual void deleteEdgeDeviceStruct( void* allEdgesDevice );

       ///  Copy all synapses' data from host to device.
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       virtual void copyEdgeHostToDevice( void* allEdgesDevice );

       ///  Copy all synapses' data from host to device.
       ///
       ///  @param  allEdgesDevice     GPU address of the allEdges struct on device memory.
       ///  @param  numVertices            Number of vertices.
       ///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
       virtual void copyEdgeHostToDevice( void* allEdgesDevice, int numVertices, int maxEdgesPerVertex );

       ///  Copy all synapses' data from device to host.
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       virtual void copyEdgeDeviceToHost( void* allEdgesDevice);

       ///  Set synapse class ID defined by enumClassSynapses for the caller's Synapse class.
       ///  The class ID will be set to classSynapses_d in device memory,
       ///  and the classSynapses_d will be referred to call a device function for the
       ///  particular synapse class.
       ///  Because we cannot use virtual function (Polymorphism) in device functions,
       ///  we use this scheme.
       ///  Note: we used to use a function pointer; however, it caused the growth_cuda crash
       ///  (see issue#137).
       virtual void setEdgeClassID();

       ///  Prints GPU SynapsesProps data.
       ///
       ///  @param  allEdgesDeviceProps   GPU address of the corresponding SynapsesDeviceProperties struct on device memory.
       virtual void printGPUEdgesProps(void* allEdgesDeviceProps) const;

   protected:
       ///  Allocate GPU memories to store all synapses' states,
       ///  and copy them from host to GPU memory.
       ///  (Helper function of allocEdgeDeviceStruct)
       ///
       ///  @param  allEdgesDevice     GPU address of the allEdges struct on device memory.
       ///  @param  numVertices            Number of vertices.
       ///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
       void allocDeviceStruct( AllDSSynapsesDeviceProperties &allEdges, int numVertices, int maxEdgesPerVertex );

       ///  Delete GPU memories.
       ///  (Helper function of deleteEdgeDeviceStruct)
       ///
       ///  @param  allEdgesDeviceProps  GPU address of the allEdges struct on device memory.
       void deleteDeviceStruct( AllDSSynapsesDeviceProperties& allEdgesDeviceProps );

       ///  Copy all synapses' data from host to device.
       ///  (Helper function of copyEdgeHostToDevice)
       ///
       ///  @param  allEdgesDevice      GPU address of the allEdges struct on device memory.
       ///  @param  allEdgesDeviceProps GPU address of the AllDSSSynapses struct on device memory.
       ///  @param  numVertices             Number of vertices.
       ///  @param  maxEdgesPerVertex   Maximum number of synapses per neuron.
       void copyHostToDevice( void* allEdgesDevice, AllDSSynapsesDeviceProperties& allEdgesDeviceProps, int numVertices, int maxEdgesPerVertex );

       ///  Copy all synapses' data from device to host.
       ///  (Helper function of copyEdgeDeviceToHost)
       ///
       ///  @param  allEdgesDeviceProps  GPU address of the allEdges struct on device memory.
       void copyDeviceToHost( AllDSSynapsesDeviceProperties& allEdgesDeviceProps);
#else // !defined(USE_GPU)
protected:
   ///  Calculate the post synapse response after a spike.
   ///
   ///  @param  iEdg        Index of the synapse to set.
   ///  @param  deltaT      Inner simulation step duration.
   virtual void changePSR(const BGSIZE iEdg, const BGFLOAT deltaT);

#endif // defined(USE_GPU)
public:

   ///  The time of the last spike.
   uint64_t *lastSpike_;

   ///  The time varying state variable \f$r\f$ for depression.
   BGFLOAT *r_;

   ///  The time varying state variable \f$u\f$ for facilitation.
   BGFLOAT *u_;

   ///  The time constant of the depression of the dynamic synapse [range=(0,10); units=sec].
   BGFLOAT *D_;

   ///  The use parameter of the dynamic synapse [range=(1e-5,1)].
   BGFLOAT *U_;

   ///  The time constant of the facilitation of the dynamic synapse [range=(0,10); units=sec].
   BGFLOAT *F_;
};

#if defined(USE_GPU)
struct AllDSSynapsesDeviceProperties : public AllSpikingSynapsesDeviceProperties
{
        ///  The time of the last spike.
        uint64_t *lastSpike_;

        ///  The time varying state variable \f$r\f$ for depression.
        BGFLOAT *r_;
        
        ///  The time varying state variable \f$u\f$ for facilitation.
        BGFLOAT *u_;
        
        ///  The time constant of the depression of the dynamic synapse [range=(0,10); units=sec].
        BGFLOAT *D_;
        
        ///  The use parameter of the dynamic synapse [range=(1e-5,1)].
        BGFLOAT *U_;

        ///  The time constant of the facilitation of the dynamic synapse [range=(0,10); units=sec].
        BGFLOAT *F_;
};
#endif // defined(USE_GPU)

