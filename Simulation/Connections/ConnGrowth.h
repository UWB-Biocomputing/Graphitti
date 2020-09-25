/**
 *      @file ConnGrowth.h
 *
 *      @brief The model of the activity dependent neurite outgrowth
 */

/**
 *
 * @class ConnGrowth ConnGrowth.h "ConnGrowth.h"
 *
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 * The activity dependent neurite outgrowth model is a phenomenological model derived by
 * a number of studies that demonstarated low level of electric activity (low firing rate)
 * stimulated neurite outgrowth, and high level of electric activity (high firing rate)
 * lead to regression (Ooyen etal. 1995).
 *
 * In this, synaptic strength (connectivity), \f$W\f$, was determined dynamically by a model of neurite
 * (cell input and output region) growth and synapse formation,
 * and a cell's region of connectivity is modeled as a circle with radius that changes
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
 * that the living preparation takes to 60,000s (approximaely 16 simulated hours).
 * Extensive analysis and simulation was performed to determine the maximum \f$\rho\f$ \f$(\rho=0.0001)\f$
 * that would not interfere with network dynamics (the increased value of \f$\rho\f$ was still
 * orders of magnitude slower than the slowest of the neuron or synapse time constants,
 * which were order of \f$10^{-2}\f$~\f$10^{-3}sec\f$).
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
 * their circla's overlap. 
 *
 * \latexonly  \subsubsection*{Credits} \endlatexonly
 * \htmlonly   <h3>Credits</h3> \endhtmlonly
 *
 * Some models in this simulator is a rewrite of CSIM (2006) and other
 * work (Stiber and Kawasaki (2007?))
 */

#pragma once

#include <iostream>
#include <vector>

#include "Connections.h"
#include "Global.h"
#include "Simulator.h"


/**
* cereal
*/
#include <cereal/types/vector.hpp>

using namespace std;

class ConnGrowth : public Connections {
public:
   ConnGrowth();

   virtual ~ConnGrowth();

   static Connections *Create() { return new ConnGrowth(); }

   /**
    *  Setup the internal structure of the class (allocate memories and initialize them).
    *
    *  @param  layout    Layout information of the neunal network.
    *  @param  neurons   The Neuron list to search from.
    *  @param  synapses  The Synapse list to search from.
    */
   virtual void setupConnections(Layout *layout, IAllNeurons *neurons, IAllSynapses *synapses);

   /**
    *  Cleanup the class (deallocate memories).
    */
   virtual void cleanupConnections();

   /**
    * Load member variables from configuration file.
    * Registered to OperationManager as Operations::op::loadParameters
    */
   virtual void loadParameters();

   /**
    *  Prints out all parameters to logging file.
    *  Registered to OperationManager as Operation::printParameters
    */
   virtual void printParameters() const;

   /**
    *  Update the connections status in every epoch.
    *
    *  @param  neurons  The Neuron list to search from.
    *  @param  layout   Layout information of the neunal network.
    *  @return true if successful, false otherwise.
    */
   virtual bool updateConnections(IAllNeurons &neurons, Layout *layout);

   /**
    *  Cereal serialization method
    *  (Serializes radii)
    */
   template<class Archive>
   void save(Archive &archive) const;

   /**
    *  Cereal deserialization method
    *  (Deserializes radii)
    */
   template<class Archive>
   void load(Archive &archive);

   /**
    *  Prints radii
    */
   void printRadii() const;

#if defined(USE_GPU)
   /**
    *  Update the weights of the Synapses in the simulation. To be clear,
    *  iterates through all source and destination neurons and updates their
    *  synaptic strengths from the weight matrix.
    *  Note: Platform Dependent.
    *
    *  @param  numNeurons          number of neurons to update.
    *  @param  neurons             the AllNeurons object.
    *  @param  synapses            the AllSynapses object.
    *  @param  m_allNeuronsDevice  Reference to the allNeurons struct in device memory.
    *  @param  m_allSynapsesDevice Reference to the allSynapses struct in device memory.
    *  @param  layout              the Layout object.
    */
   virtual void updateSynapsesWeights(const int numNeurons,
         IAllNeurons &neurons, IAllSynapses &synapses,
         AllSpikingNeuronsDeviceProperties* m_allNeuronsDevice,
         AllSpikingSynapsesDeviceProperties* m_allSynapsesDevice,
         Layout *layout);
#else
   /**
    *  Update the weights of the Synapses in the simulation. To be clear,
    *  iterates through all source and destination neurons and updates their
    *  synaptic strengths from the weight matrix.
    *  Note: Platform Dependent.
    *
    *  @param  numNeurons Number of neurons to update.
    *  @param  ineurons    the AllNeurons object.
    *  @param  isynapses   the AllSynapses object.
    *  @param  layout      the Layout object.
    * 
    */
   virtual void
   updateSynapsesWeights(const int numNeurons,
         IAllNeurons &neurons,
         IAllSynapses &synapses,
         Layout *layout);

#endif
private:
   /**
    *  Calculates firing rates, neuron radii change and assign new values.
    *
    *  @param  neurons  The Neuron list to search from.
    */
   void updateConns(IAllNeurons &neurons);

   /**
    *  Update the distance between frontiers of Neurons.
    *
    *  @param  numNeurons Number of neurons to update.
    *  @param  layout      Layout information of the neunal network.
    */
   void updateFrontiers(const int numNeurons, Layout *layout);

   /**
    *  Update the areas of overlap in between Neurons.
    *
    *  @param  numNeurons Number of Neurons to update.
    *  @param  layout      Layout information of the neunal network.
    */
   void updateOverlap(BGFLOAT numNeurons, Layout *layout);

public:
   struct GrowthParams {
      BGFLOAT epsilon;   //null firing rate(zero outgrowth)
      BGFLOAT beta;      //sensitivity of outgrowth to firing rate
      BGFLOAT rho;       //outgrowth rate constant
      BGFLOAT targetRate; // Spikes/second
      BGFLOAT maxRate;   // = targetRate / epsilon;
      BGFLOAT minRadius; // To ensure that even rapidly-firing neurons will connect to
      // other neurons, when within their RFS.
      BGFLOAT startRadius; // No need to wait a long time before RFs start to overlap
   };

   //! structure to keep growth parameters
   GrowthParams growthParams_;

   //! spike count for each epoch
   int *spikeCounts_;

   //! radii size （2020/2/13 add radiiSize for use in serialization/deserialization)
   int radiiSize_;

   //! synapse weight
   CompleteMatrix *W_;

   //! neuron radii
   VectorMatrix *radii_;

   //! spiking rate
   VectorMatrix *rates_;

   //! distance between connection frontiers
   CompleteMatrix *delta_;

   //! areas of overlap
   CompleteMatrix *area_;

   //! neuron's outgrowth
   VectorMatrix *outgrowth_;

   //! displacement of neuron radii
   VectorMatrix *deltaR_;

};

/**
 *  Cereal serialization method
 *  (Serializes radii)
 */
template<class Archive>
void ConnGrowth::save(Archive &archive) const {
   // uses vector to save radii
   vector<BGFLOAT> radiiVector;
   for (int i = 0; i < radiiSize_; i++) {
      radiiVector.push_back((*radii_)[i]);
   }
   // serialization
   archive(radiiVector);
}

/**
 *  Cereal deserialization method
 *  (Deserializes radii)
 */
template<class Archive>
void ConnGrowth::load(Archive &archive) {
   // uses vector to load radii
   vector<BGFLOAT> radiiVector;

   // deserializing data to this vector
   archive(radiiVector);

   // check to see if serialized data size matches object size
   if (radiiVector.size() != radiiSize_) {
      cerr << "Failed deserializing radii. Please verify totalNeurons data member." << endl;
      throw cereal::Exception("Deserialization Error");
   }

   // assigns serialized data to objects
   for (int i = 0; i < radiiSize_; i++) {
      (*radii_)[i] = radiiVector[i];
   }
}
