/**
 * @file ConnGrowth.h
 * 
 * @ingroup Simulator/Connections
 *
 * @brief The model of the activity-dependent neurite outgrowth
 *
 * The activity-dependent neurite outgrowth model is a phenomenological model derived by
 * a number of studies that demonstrated a low-level of electric activity (low firing rate)
 * stimulated neurite outgrowth, and high level of electric activity (high firing rate)
 * lead to regression (Ooyen et al. 1995).
 *
 * In this, synaptic strength (connectivity), \f$W\f$, was determined dynamically by a model of neurite
 * (cell input and output region) growth and synapse formation,
 * and a cell's region of connectivity is modeled as a circle with a radius that changes
 * at a rate inversely proportional to a sigmoidal function of cell firing rate:
 * \f[
 *  \frac{d R_{i}}{dt} = \rho G(F_{i})
 * \f]
 * \f[
 *  G(F_{i}) = 1 - \frac{2}{1 + exp((\epsilon - F_{i}) / \beta)}
 * \f]
 * where \f$R_{i}\f$ is the radius of connectivity of neuron \f$i\f$, \f$F_{i}\f$ is neuron i's firing rate
 * (normalized to be in the range \f$[0,1]\f$, \f$\rho\f$ is an outgrowth rate constant, \f$\epsilon\f$ is a constant
 * that sets the "null point" for outgrowth (the firing rate in spikes/sec that causes
 * no outgrowth or retration), and \f$\beta\f$ determines the slope of \f$G(\cdot)\f$.
 * One divergence in these simulations from strict modeling of the living preparation
 * was that \f$\rho\f$ was increased to reduce simulated development times from the weeks
 * that the living preparation takes 60,000s (approximately 16 simulated hours).
 * Extensive analysis and simulation were performed to determine the maximum \f$\rho\f$ \f$(\rho=0.0001)\f$
 * that would not interfere with network dynamics (the increased value of \f$\rho\f$ was still
 * orders of magnitude slower than the slowest of the neuron or synapse time constants,
 * which were the order of \f$10^{-2}\f$~\f$10^{-3}sec\f$).
 *
 * Synaptic strengths were computed for all pairs of neurons that had overlapping connectivity
 * regions as the area of their circle's overlap:
 * \f[
 *  r_0^2 = r_1^2 + |AB|^2 - 2 r_1 |AB| cos(\angle CBA)
 * \f]
 * \f[
 *  cos(\angle CBA) = \frac{r_1^2 + |AB|^2 - r_0^2}{2 r_1 |AB|}
 * \f]
 * \f[
 *  \angle CBD =  2 \angle CBA
 * \f]
 * \f[
 *  cos(\angle CAB) = \frac{r_0^2 + |AB|^2 - r_1^2}{2 r_0 |AB|}
 * \f]
 * \f[
 *  \angle CAD =  2 \angle CAB
 * \f]
 * \f[
 *  w_{01} = \frac{1}{2} \angle CBD r_1^2 - \frac{1}{2} r_1^2 sin(\angle CBD) + \frac{1}{2} \angle CAD r_0^2 - \frac{1}{2} r_0^2 sin(\angle CAD)
 * \f]
 * \f[
 *  w_{01} = w_{10}
 * \f]
 * where A and B are the locations of neurons A and B, \f$r_0\f$ and 
 * \f$r_1\f$ are the neurite radii of neuron A and B, C and B are locations of intersections 
 * of neurite boundaries of neuron A and B, and \f$w_{01}\f$ and \f$w_{10}\f$ are the areas of 
 * their circle's overlap. 
 *
 * Some models in this simulator is a rewrite of CSIM (2006) and other
 * work (Stiber and Kawasaki (2007?))
 * 
 * NOTE: Currently ConnGrowth doesn't create edges ad the beginning of the simulation.
 */

#pragma once

#include "Connections.h"
#include "Global.h"
#include "Simulator.h"
#include <iostream>
#include <vector>
// cereal
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>

using namespace std;

class ConnGrowth : public Connections {
public:
   ConnGrowth();

   virtual ~ConnGrowth() = default;

   static Connections *Create()
   {
      return new ConnGrowth();
   }

   ///  Setup the internal structure of the class (allocate memories and initialize them).
   virtual void setup() override;

   /// Load member variables from configuration file.
   /// Registered to OperationManager as Operations::op::loadParameters
   virtual void loadParameters() override;

   /// Registers history variables for recording during simulation
   virtual void registerHistoryVariables() override;

   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const override;

   ///  Update the connections status in every epoch.
   ///
   ///  @param  vertices  The vertex list to search from.
   ///  @return true if successful, false otherwise.
   virtual bool updateConnections(AllVertices &vertices) override;

   ///  Cereal serialization method
   template <class Archive> void serialize(Archive &archive);

   ///  Prints radii
   void printRadii() const;

#if defined(USE_GPU)
   ///  Update the weights of the Synapses in the simulation. To be clear,
   ///  iterates through all source and destination neurons and updates their
   ///  synaptic strengths from the weight matrix.
   ///  Note: Platform Dependent.
   ///
   ///  @param  numVertices          The number of vertices to update.
   ///  @param  vertices             The AllVertices object.
   ///  @param  edges                The AllEdges object.
   ///  @param  allVerticesDevice    GPU address of the AllVertices struct in device memory.
   ///  @param  allEdgesDevice       GPU address of the AllEdges struct in device memory.
   ///  @param  layout               The Layout object.
   virtual void updateEdgesWeights(int numVertices, AllVertices &vertices, AllEdges &edges,
                                   AllVerticesDeviceProperties *allVerticesDevice,
                                   AllEdgesDeviceProperties *allEdgesDevice,
                                   Layout &layout) override;
#else
   ///  Update the weights of the Synapses in the simulation. To be clear,
   ///  iterates through all source and destination neurons and updates their
   ///  synaptic strengths from the weight matrix.
   ///  Note: Platform Dependent.
   virtual void updateEdgesWeights() override;

#endif
private:
   ///  Calculates firing rates, neuron radii change and assign new values.
   ///
   ///  @param  neurons  The Neuron list to search from.
   void updateConns(AllVertices &neurons);

   /// Update the distance between frontiers of Neurons.
   void updateFrontiers();

   ///  Update the areas of overlap in between Neurons.
   void updateOverlap();

public:
   struct GrowthParams {
      BGFLOAT epsilon;       ///< null firing rate(zero outgrowth)
      BGFLOAT beta;          ///< sensitivity of outgrowth to firing rate
      BGFLOAT rho;           ///< outgrowth rate constant
      BGFLOAT targetRate;    ///<  Spikes/second
      BGFLOAT maxRate;       ///<  = targetRate / epsilon;
      BGFLOAT minRadius;     ///<  To ensure that even rapidly-firing neurons will connect to
                             ///< other neurons, when within their RFS.
      BGFLOAT startRadius;   ///< No need to wait a long time before RFs start to overlap

      ///  Cereal serialization method
      template <class Archive> void serialize(Archive &archive)
      {
         archive(cereal::make_nvp("epsilon", epsilon), cereal::make_nvp("beta", beta),
                 cereal::make_nvp("rho", rho), cereal::make_nvp("targetRate", targetRate),
                 cereal::make_nvp("maxRate", maxRate), cereal::make_nvp("minRadius", minRadius),
                 cereal::make_nvp("startRadius", startRadius));
      }
   };

   /// structure to keep growth parameters
   GrowthParams growthParams_;

   /// radii size
   int radiiSize_;

   /// synapse weight
   CompleteMatrix W_;

   /// neuron radii
   VectorMatrix radii_;

   /// spiking rate
   VectorMatrix rates_;

   /// distance between connection frontiers
   CompleteMatrix delta_;

   /// areas of overlap
   CompleteMatrix area_;

   /// neuron's outgrowth
   VectorMatrix outgrowth_;

   /// displacement of neuron radii
   VectorMatrix deltaR_;
};

CEREAL_REGISTER_TYPE(ConnGrowth);   // to enable polymorphism

///  Cereal serialization method
template <class Archive> void ConnGrowth::serialize(Archive &archive)
{
   archive(cereal::base_class<Connections>(this), cereal::make_nvp("radiiSize", radiiSize_),
           cereal::make_nvp("growthParams", growthParams_), cereal::make_nvp("W", W_),
           cereal::make_nvp("radii", radii_), cereal::make_nvp("rates", rates_),
           cereal::make_nvp("delta", delta_), cereal::make_nvp("area", area_),
           cereal::make_nvp("outgrowth", outgrowth_), cereal::make_nvp("deltaR", deltaR_));
}
