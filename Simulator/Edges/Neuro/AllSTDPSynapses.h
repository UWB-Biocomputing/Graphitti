/**
 *  @file AllSTDPSynapses.h
 * 
 *  @ingroup Simulator/Edges
 *
 *  @brief A container of all STDP synapse data
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
 *  Implements the basic weight update for a time difference \f$Delta =
 *  t_{post}-t_{pre}\f$ with presynaptic spike at time \f$t_{pre}\f$ and
 *  postsynaptic spike at time \f$t_{post}\f$. Then, the weight update is given by
 *  \f$dw =  Apos_ * exp(-Delta/taupos_)\f$ for \f$Delta > 0\f$, and \f$dw =  Aneg_ *
 *  exp(-Delta/tauneg_)\f$ for \f$Delta < 0\f$. (set \f$useFroemkeDanSTDP_=0\f$ and
 *  \f$mupos_=muneg_=0\f$ for this basic update rule).
 *  
 *  It is also possible to use an
 *  extended multiplicative update by changing mupos_ and muneg_. Then \f$dw =
 *  (Wex_-W)^{mupos_} * Apos_ * exp(-Delta/taupos_)\f$ for \f$Delta > 0\f$ and \f$dw =
 *  W^{mupos_} * Aneg_ * exp(Delta/tauneg_)\f$ for \f$Delta < 0\f$. (see Guetig,
 *  Aharonov, Rotter and Sompolinsky (2003). Learning input correlations through
 *  non-linear asymmetric Hebbian plasticity. Journal of Neuroscience 23.
 *  pp.3697-3714.)
 *      
 *  Set \f$useFroemkeDanSTDP_=1\f$ (this is the default value) and
 *  use \f$tauspost_\f$ and \f$tauspre_\f$ for the rule given in Froemke and Dan
 *  (2002). Spike-timing-dependent synaptic modification induced by natural spike
 *  trains. Nature 416 (3/2002). 
 * 
 *  05/01/2020
 *  Changed the default weight update rule and all formula constants using the 
 *  independent model (a basic STDP model) and multiplicative model in
 *  Froemke and Dan (2002). Spike-timing-dependent synaptic modification induced by natural spike
 *  trains. Nature 416 (3/2002)
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

#include "AllSpikingSynapses.h"

struct AllSTDPSynapsesDeviceProperties;

class AllSTDPSynapses : public AllSpikingSynapses {
public:
   AllSTDPSynapses();

   AllSTDPSynapses(const int numVertices, const int maxEdges);

   virtual ~AllSTDPSynapses();

   static AllEdges *Create() { return new AllSTDPSynapses(); }

   ///  Setup the internal structure of the class (allocate memories and initialize them).
   virtual void setupEdges() override;

   ///  Reset time varying state vars and recompute decay.
   ///
   ///  @param  iEdg     Index of the synapse to set.
   ///  @param  deltaT   Inner simulation step duration
   virtual void resetEdge(const BGSIZE iEdg, const BGFLOAT deltaT) override;

   ///  Check if the back propagation (notify a spike event to the pre neuron)
   ///  is allowed in the synapse class.
   ///
   ///  @return true if the back propagation is allowed.
   virtual bool allowBackPropagation() override;

   /// Load member variables from configuration file. Registered to OperationManager as
   /// Operation::op::loadParameters
   virtual void loadParameters() override;
   
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

   ///  Initializes the queues for the Synapse.
   ///
   ///  @param  iEdg   index of the synapse to set.
   virtual void initSpikeQueue(const BGSIZE iEdg) override;

#if defined(USE_GPU)
   public:
       ///  Allocate GPU memories to store all synapses' states,
       ///  and copy them from host to GPU memory.
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       virtual void allocEdgeDeviceStruct( void** allEdgesDevice ) override;

       ///  Allocate GPU memories to store all synapses' states,
       ///  and copy them from host to GPU memory.
       ///
       ///  @param  allEdgesDevice     GPU address of the allEdges struct on device memory.
       ///  @param  numVertices            Number of vertices.
       ///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
       virtual void allocEdgeDeviceStruct( void** allEdgesDevice, int numVertices, int maxEdgesPerVertex ) override;

       ///  Delete GPU memories.
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       virtual void deleteEdgeDeviceStruct( void* allEdgesDevice ) override;

       ///  Copy all synapses' data from host to device.
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       virtual void copyEdgeHostToDevice( void* allEdgesDevice ) override;

       ///  Copy all synapses' data from host to device.
       ///
       ///  @param  allEdgesDevice     GPU address of the allEdges struct on device memory.
       ///  @param  numVertices            Number of vertices.
       ///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
       virtual void copyEdgeHostToDevice( void* allEdgesDevice, int numVertices, int maxEdgesPerVertex ) override;

       ///  Copy all synapses' data from device to host.
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       virtual void copyEdgeDeviceToHost( void* allEdgesDevice ) override;

       ///  Advance all the Synapses in the simulation.
       ///  Update the state of all synapses for a time step.
       ///
       ///  @param  allEdgesDevice      GPU address of the allEdges struct on device memory.
       ///  @param  allVerticesDevice       GPU address of the allNeurons struct on device memory.
       ///  @param  edgeIndexMapDevice  GPU address of the EdgeIndexMap on device memory.
       virtual void advanceEdges( void* allEdgesDevice, void* allVerticesDevice, void* edgeIndexMapDevice ) override;

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
       virtual void printGPUEdgesProps( void* allEdgesDeviceProps ) const override;

   protected:
       ///  Allocate GPU memories to store all synapses' states,
       ///  and copy them from host to GPU memory.
       ///  (Helper function of allocEdgeDeviceStruct)
       ///
       ///  @param  allEdgesDevice     GPU address of the allEdges struct on device memory.
       ///  @param  numVertices            Number of vertices.
       ///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
       void allocDeviceStruct( AllSTDPSynapsesDeviceProperties &allEdgesDevice, int numVertices, int maxEdgesPerVertex );

       ///  Delete GPU memories.
       ///  (Helper function of deleteEdgeDeviceStruct)
       ///
       ///  @param  allEdgesDevice  GPU address of the allEdges struct on device memory.
       void deleteDeviceStruct( AllSTDPSynapsesDeviceProperties& allEdgesDevice );

       ///  Copy all synapses' data from host to device.
       ///  (Helper function of copyEdgeHostToDevice)
       ///
       ///  @param  allEdgesDevice       GPU address of the allEdges struct on device memory.
       ///  @param  allEdgesDeviceProps  GPU address of the corresponding SynapsesDeviceProperties struct on device memory.
       ///  @param  numVertices              Number of vertices.
       ///  @param  maxEdgesPerVertex    Maximum number of synapses per neuron.
       void copyHostToDevice( void* allEdgesDevice, AllSTDPSynapsesDeviceProperties& allEdgesDeviceProps, int numVertices, int maxEdgesPerVertex );

       ///  Copy all synapses' data from device to host.
       ///  (Helper function of copyEdgeDeviceToHost)
       ///
       ///  @param  allEdgesDevice     GPU address of the allEdges struct on device memory.
       ///  @param  numVertices            Number of vertices.
       ///  @param  maxEdgesPerVertex  Maximum number of synapses per neuron.
       void copyDeviceToHost( AllSTDPSynapsesDeviceProperties& allEdgesDevice );
#else // !defined(USE_GPU)
public:
   ///  Advance one specific Synapse.
   ///  Update the state of synapse for a time step
   ///
   ///  @param  iEdg      Index of the Synapse to connect to.
   ///  @param  neurons   The Neuron list to search from.
   virtual void advanceEdge(const BGSIZE iEdg, AllVertices *neurons) override;

   ///  Prepares Synapse for a spike hit (for back propagation).
   ///
   ///  @param  iEdg   Index of the Synapse to connect to.
   virtual void postSpikeHit(const BGSIZE iEdg) override;

protected:
   ///  Checks if there is an input spike in the queue (for back propagation).
   ///
   ///  @param  iEdg   Index of the Synapse to connect to.
   ///  @return true if there is an input spike event.
   bool isSpikeQueuePost(const BGSIZE iEdg);

   virtual BGFLOAT synapticWeightModification(const BGSIZE iEdg, BGFLOAT edgeWeight, double delta);

private:
   ///  Adjust synapse weight according to the Spike-timing-dependent synaptic modification
   ///  induced by natural spike trains
   ///
   ///  @param  iEdg        Index of the synapse to set.
   ///  @param  delta       Pre/post synaptic spike interval.
   ///  @param  epost       Params for the rule given in Froemke and Dan (2002).
   ///  @param  epre        Params for the rule given in Froemke and Dan (2002).
   ///  @param srcVertex Index of source neuron
   ///  @param destVertex Index of destination neuron
   void stdpLearning(const BGSIZE iEdg, double delta, double epost, double epre,
                     int srcVertex, int destVertex);

#endif
public:
   ///  The synaptic transmission delay (delay of dendritic backpropagating spike),
   ///  descretized into time steps.
   int *totalDelayPost_;

   ///  Pointer to the delayed queue
   uint32_t *delayQueuePost_;

   ///  The index indicating the current time slot in the delayed queue.
   int *delayIndexPost_;

   ///  Length of the delayed queue.
   int *delayQueuePostLength_;

   ///  Used for extended rule by Froemke and Dan. See Froemke and Dan (2002).
   ///  Spike-timing-dependent synaptic modification induced by natural spike trains.
   ///  Nature 416 (3/2002).
   BGFLOAT *tauspost_;

   ///  sed for extended rule by Froemke and Dan.
   BGFLOAT *tauspre_;

   ///  Timeconstant of exponential decay of positive learning window for STDP.
   BGFLOAT *taupos_;

   ///  Timeconstant of exponential decay of negative learning window for STDP.
   BGFLOAT *tauneg_;

   ///  No learning is performed if \f$|Delta| = |t_{post}-t_{pre}| < STDPgap_\f$
   BGFLOAT *STDPgap_;

   ///  The maximal/minimal weight of the synapse [readwrite; units=;]
   BGFLOAT *Wex_;

   ///  Defines the peak of the negative exponential learning window.
   BGFLOAT *Aneg_;

   ///  Defines the peak of the positive exponential learning window.
   BGFLOAT *Apos_;

   ///  Extended multiplicative positive update:
   ///  \f$dw = (Wex_-W)^{mupos_} * Apos_ * exp(-Delta/taupos_)\f$.
   ///  Set to 0 for basic update. See Guetig, Aharonov, Rotter and Sompolinsky (2003).
   ///  Learning input correlations through non-linear asymmetric Hebbian plasticity.
   ///  Journal of Neuroscience 23. pp.3697-3714.
   BGFLOAT *mupos_;

   ///  Extended multiplicative negative update:
   ///  \f$dw = W^{mupos_} * Aneg_ * exp(Delta/tauneg_)\f$. Set to 0 for basic update.
   BGFLOAT *muneg_;


   BGFLOAT defaultSTDPgap_;
   BGFLOAT tauspost_I_;
   BGFLOAT tauspre_I_;
   BGFLOAT tauspost_E_;
   BGFLOAT tauspre_E_;
   BGFLOAT taupos_I_;
   BGFLOAT tauneg_I_;
   BGFLOAT taupos_E_;
   BGFLOAT tauneg_E_;
   BGFLOAT Wex_I_;
   BGFLOAT Wex_E_;
   BGFLOAT Aneg_I_;
   BGFLOAT Aneg_E_;
   BGFLOAT Apos_I_;
   BGFLOAT Apos_E_;
};

#if defined(USE_GPU)
struct AllSTDPSynapsesDeviceProperties : public AllSpikingSynapsesDeviceProperties
{
        ///  The synaptic transmission delay (delay of dendritic backpropagating spike), 
        ///  descretized into time steps.
        int *totalDelayPost_;

        ///  Pointer to the delayed queue
        uint32_t *delayQueuePost_;

        ///  The index indicating the current time slot in the delayed queue.
        int *delayIndexPost_;

        ///  Length of the delayed queue.
        int *delayQueuePostLength_;

        ///  Used for extended rule by Froemke and Dan. See Froemke and Dan (2002). 
        ///  Spike-timing-dependent synaptic modification induced by natural spike trains. 
        ///  Nature 416 (3/2002).
        BGFLOAT *tauspost_;

        ///  sed for extended rule by Froemke and Dan.
        BGFLOAT *tauspre_;

        ///  Timeconstant of exponential decay of positive learning window for STDP.
        BGFLOAT *taupos_;

        ///  Timeconstant of exponential decay of negative learning window for STDP.
        BGFLOAT *tauneg_;

        ///  No learning is performed if \f$|Delta| = |t_{post}-t_{pre}| < STDPgap_\f$
        BGFLOAT *STDPgap_;

        ///  The maximal/minimal weight of the synapse [readwrite; units=;]
        BGFLOAT *Wex_;

        ///  Defines the peak of the negative exponential learning window.
        BGFLOAT *Aneg_;

        ///  Defines the peak of the positive exponential learning window.
        BGFLOAT *Apos_;

        ///  Extended multiplicative positive update: 
        ///  \f$dw = (Wex_-W)^{mupos_} * Apos_ * exp(-Delta/taupos_)\f$. 
        ///  Set to 0 for basic update. See Guetig, Aharonov, Rotter and Sompolinsky (2003). 
        ///  Learning input correlations through non-linear asymmetric Hebbian plasticity. 
        ///  Journal of Neuroscience 23. pp.3697-3714.
        BGFLOAT *mupos_;

        ///  Extended multiplicative negative update: 
        ///  \f$dw = W^{mupos_} * Aneg_ * exp(Delta/tauneg_)\f$. Set to 0 for basic update.
        BGFLOAT *muneg_;
  
        ///  True if use the rule given in Froemke and Dan (2002).
        bool *useFroemkeDanSTDP_;
};
#endif // defined(USE_GPU)
