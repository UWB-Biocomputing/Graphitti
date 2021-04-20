/**
 * @file AllDynamicSTDPSynapses.h
 * 
 * @ingroup Simulator/Edges
 *
 * @brief A container of all dynamic STDP synapse data
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
 *  The AllDynamicSTDPSynapses inherited properties from the AllDSSynapses and the AllSTDPSynapses
 *  classes (multiple inheritance), and both the AllDSSynapses and the AllSTDPSynapses classes are
 *  the subclass of the AllSpikingSynapses class. Therefore, this is known as a diamond class
 *  inheritance, which causes the problem of ambibuous hierarchy compositon. To solve the
 *  problem, we can use the virtual inheritance. 
 *  However, the virtual inheritance will bring another problem. That is, we cannot static cast
 *  from a pointer to the AllEdges class to a pointer to the AllDSSynapses or the AllSTDPSynapses 
 *  classes. Compiler requires dynamic casting because vtable mechanism is involed in solving the 
 *  casting. But the dynamic casting cannot be used for the pointers to device (CUDA) memories. 
 *  Considering these issues, I decided that making the AllDynamicSTDPSynapses class the subclass
 *  of the AllSTDPSynapses class and adding properties of the AllDSSynapses class to it (fumik).
 *   
 *  Independent model:
 *  \f$Delta = t_{post}-t_{pre}\f$ with presynaptic spike at time \f$t_{pre}\f$ and
 *  postsynaptic spike at time \f$t_{post}\f$. Then, the weight update is given by
 *  \f$dw =  Apos_ * exp(-Delta/taupos_)\f$ for \f$Delta > 0\f$, and \f$dw =  Aneg_ *
 *  exp(-Delta/tauneg_)\f$ for \f$Delta < 0\f$. dw is the percentage change in synaptic weight.
 *  (set \f$useFroemkeDanSTDP_=false\f$ and \f$mupos_=muneg_=0\f$ for this basic update rule).
 *  
 *  Multiplicative model:
 *  \f$dw = 1.0 + dw * epre * epost\f$ dw is percent change, so adding 1.0 become the scale ratio
 *  \f$W = W * dw\f$ multiply dw (scale ratio) to the current weight to get the new weight
 *  
 *  Note1:This time we don't use useFroemkeDanSTDP_ (useFroemkeDanSTDP_= false) and mupos_ and muneg_ (mupos_=muneg_=0)
 *  Note2:Based on the FroemkeDan paper, the STDP learning rule only applies to excititory synapses, so we
 *  implement it to have only excititory neurons do STDP weight adjustment 
 */

#pragma once

#include "AllSTDPSynapses.h"

struct AllDynamicSTDPSynapsesDeviceProperties;

class AllDynamicSTDPSynapses : public AllSTDPSynapses {
public:
   AllDynamicSTDPSynapses();

   AllDynamicSTDPSynapses(const int numVertices, const int maxEdges);

   virtual ~AllDynamicSTDPSynapses();

   static AllEdges *Create() { return new AllDynamicSTDPSynapses(); }

   ///  Setup the internal structure of the class (allocate memories and initialize them).
   virtual void setupEdges() override;

   ///  Reset time varying state vars and recompute decay.
   ///
   ///  @param  iEdg     Index of the synapse to set.
   ///  @param  deltaT   Inner simulation step duration
   virtual void resetEdge(const BGSIZE iEdg, const BGFLOAT deltaT) override;

   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const override;

   ///  Create a Synapse and connect it to the model.
   ///
   ///  @param  iEdg        Index of the synapse to set.
   ///  @param  srcVertex   Coordinates of the source Neuron.
   ///  @param  destVertex  Coordinates of the destination Neuron.
   ///  @param  sumPoint    Summation point address.
   ///  @param  deltaT      Inner simulation step duration.
   ///  @param  type        Type of the Synapse to create.
   virtual void createEdge(const BGSIZE iEdg, int srcVertex, int destVertex, BGFLOAT *sumPoint, const BGFLOAT deltaT,
                              edgeType type) override;

   ///  Prints SynapsesProps data.
   virtual void printSynapsesProps() const override;

protected:
   ///  Setup the internal structure of the class (allocate memories and initialize them).
   ///
   ///  @param  numVertices   Total number of vertices in the network.
   ///  @param  maxEdges  Maximum number of synapses per neuron.
   virtual void setupEdges(const int numVertices, const int maxEdges) override;

   ///  Sets the data for Synapse to input's data.
   ///
   ///  @param  input  istream to read from.
   ///  @param  iEdg   Index of the synapse to set.
   virtual void readEdge(istream &input, const BGSIZE iEdg) override;

   ///  Write the synapse data to the stream.
   ///
   ///  @param  output  stream to print out to.
   ///  @param  iEdg    Index of the synapse to print out.
   virtual void writeEdge(ostream &output, const BGSIZE iEdg) const override;

#if defined(USE_GPU)
   public:
       ///  Allocate GPU memories to store all synapses' states,
       ///  and copy them from host to GPU memory.
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       virtual void allocEdgeDeviceStruct( void** allEdgesDevice) override;

       ///  Allocate GPU memories to store all synapses' states,
       ///  and copy them from host to GPU memory.
       ///
       ///  @param  allEdgesDevice     GPU address of the allEdges struct on device memory.
       ///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
       virtual void allocEdgeDeviceStruct(void** allEdgesDevice, int numVertices, int maxEdgesPerVertex) override;

       ///  Delete GPU memories.
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       virtual void deleteEdgeDeviceStruct(void* allEdgesDevice) override;

       ///  Copy all synapses' data from host to device.
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       virtual void copyEdgeHostToDevice(void* allEdgesDevice) override;

       ///  Copy all synapses' data from host to device.
       ///
       ///  @param  allEdgesDevice     GPU address of the allEdges struct on device memory.
       ///  @param  numVertices            Number of vertices.
       ///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
       virtual void copyEdgeHostToDevice( void* allEdgesDevice, int numVertices, int maxEdgesPerVertex ) override;

       ///  Copy all synapses' data from device to host.
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       virtual void copyEdgeDeviceToHost(void* allEdgesDevice) override;

       ///  Set synapse class ID defined by enumClassSynapses for the caller's Synapse class.
       ///  The class ID will be set to classSynapses_d in device memory,
       ///  and the classSynapses_d will be referred to call a device function for the
       ///  particular synapse class.
       ///  Because we cannot use virtual function (Polymorphism) in device functions,
       ///  we use this scheme.
       ///  Note: we used to use a function pointer; however, it caused the growth_cuda crash
       ///  (see issue#137).
       virtual void setEdgeClassID() override;

       ///  Prints GPU SynapsesProps data.
       ///
       ///  @param  allEdgesDeviceProps   GPU address of the corresponding SynapsesDeviceProperties struct on device memory.
       virtual void printGPUEdgesProps(void* allEdgesDeviceProps) const override;

   protected:
       ///  Allocate GPU memories to store all synapses' states,
       ///  and copy them from host to GPU memory.
       ///  (Helper function of allocEdgeDeviceStruct)
       ///
       ///  @param  allEdgesDevice     GPU address of the allEdges struct on device memory.
       ///  @param  numVertices            Number of vertices.
       ///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
       void allocDeviceStruct( AllDynamicSTDPSynapsesDeviceProperties &allEdges, int numVertices, int maxEdgesPerVertex );

       ///  Delete GPU memories.
       ///  (Helper function of deleteEdgeDeviceStruct)
       ///
       ///  @param  allEdgesDeviceProps  GPU address of the allEdges struct on device memory.
       void deleteDeviceStruct( AllDynamicSTDPSynapsesDeviceProperties& allEdgesDeviceProps );

       ///  Copy all synapses' data from host to device.
       ///  (Helper function of copyEdgeHostToDevice)
       ///
       ///  @param  allEdgesDevice         GPU address of the allEdges struct on device memory.
       ///  @param  allEdgesDeviceProps    GPU address of the allDynamicSTDPSSynapses struct on device memory.
       ///  @param  numVertices                Number of vertices.
       ///  @param  maxEdgesPerVertex      Maximum number of synapses per neuron.
       void copyHostToDevice( void* allEdgesDevice, AllDynamicSTDPSynapsesDeviceProperties& allEdgesDeviceProps, int numVertices, int maxEdgesPerVertex );

       ///  Copy all synapses' data from device to host.
       ///  (Helper function of copyEdgeDeviceToHost)
       ///
       ///  @param  allEdgesDeviceProps     GPU address of the allDynamicSTDPSSynapses struct on device memory.
       ///  @param  numVertices                 Number of vertices.
       ///  @param  maxEdgesPerVertex       Maximum number of synapses per neuron.
       void copyDeviceToHost(AllDynamicSTDPSynapsesDeviceProperties& allEdgesDeviceProps);
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
struct AllDynamicSTDPSynapsesDeviceProperties : public AllSTDPSynapsesDeviceProperties
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

