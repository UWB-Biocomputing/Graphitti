/**
 * @file AllLIFNeurons.h
 * 
 * @ingroup Simulator/Vertices
 *
 * @brief A container of all LIF neuron data
 *
 * A container of all spiking neuron data.
 * This is the base class of all spiking neuron classes.
 *
 * The class uses a data-centric structure, which utilizes a structure as the containers of
 * all neuron.
 *
 * The container holds neuron parameters of all neurons.
 * Each kind of neuron parameter is stored in a 1D array, of which length
 * is number of all neurons. Each array of a neuron parameter is pointed by a
 * corresponding member variable of the neuron parameter in the class.
 *
 * This structure was originally designed for the GPU implementation of the
 * simulator, and this refactored version of the simulator simply uses that design for
 * all other implementations as well. This is to simplify transitioning from
 * single-threaded to multi-threaded.
 *
 * A standard leaky-integrate-and-fire neuron model is implemented
 * where the membrane potential \f$V_m\f$ of a neuron is given by
 * \f[
 *   \tau_m \frac{d V_m}{dt} = -(V_m-V_{resting}) + R_m \cdot (I_{syn}(t)+I_{inject}+I_{noise})
 * \f]
 * where \f$\tau_m=C_m\cdot R_m\f$ is the membrane time constant,
 * \f$R_m\f$ is the membrane resistance, \f$I_{syn}(t)\f$ is the
 * current supplied by the synapses, \f$I_{inject}\f$ is a
 * non-specific background current and \f$I_{noise}\f$ is a
 * Gaussian random variable with zero mean and a given variance
 * noise.
 *
 * At time \f$t=0\f$ \f$V_m\f$ is set to \f$V_{init}\f$. If
 * \f$V_m\f$ exceeds the threshold voltage \f$V_{thresh}\f$ it is
 * reset to \f$V_{reset}\f$ and hold there for the length
 * \f$T_{refract}\f$ of the absolute refractory period.
 *
 * The exponential Euler method is used for numerical integration.
 * The main idea behind the exponential Euler rule is that many biological 
 *mprocesses are governed by an exponential decay function.
 * For an equation of the form:
 * \f[
 *  \frac{d y}{dt} = A - By
 * \f]
 *
 * its scheme is given by:
 * \f[
 *  y(t+\Delta t) = y(t) \cdot \mathrm{e}^{-B \Delta t} + \frac{A}{B} \cdot (1 - \mathrm{e}^{-B \Delta t})
 * \f]
 *
 * After appropriate substituting all variables:
 * \f[
 *  y(t) = V_m(t)
 * \f]
 * \f[
 *  A = \frac{1}{\tau_m} \cdot (R_m \cdot (I_{syn}(t)+I_{inject}+I_{noise}) + V_{resting})
 * \f]
 * and
 * \f[
 *  B = \frac{1}{\tau_m}
 * \f]
 * 
 * we obtain the exponential Euler step:
 * \f[
 *  V_m(t+\Delta t) = C1 \cdot V_m(t) + 
 *  C2 \cdot (I_{syn}(t)+I_{inject}+I_{noise}+\frac{V_{resting}}{R_m}) 
 * \f]
 * where \f$C1 = \mathrm{e}^{-\frac{\Delta t}{\tau_m}}\f$ and 
 * \f$C2 = R_m \cdot (1 - \mathrm{e}^{-\frac{\Delta t}{\tau_m}})\f$.
 */
#pragma once

#include "Global.h"
#include "AllIFNeurons.h"
#include "AllSpikingSynapses.h"

// Class to hold all data necessary for all the Neurons.
class AllLIFNeurons : public AllIFNeurons {
public:

   AllLIFNeurons();

   virtual ~AllLIFNeurons();

   ///  Creates an instance of the class.
   ///
   ///  @return Reference to the instance of the class.
   static AllVertices *Create() { return new AllLIFNeurons(); }

   ///  Prints out all parameters of the neurons to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const override;

#if defined(USE_GPU)
   public:

       ///  Update the state of all neurons for a time step
       ///  Notify outgoing synapses if neuron has fired.
       ///
       ///  @param  synapses               Reference to the allEdges struct on host memory.
       ///  @param  allVerticesDevice       GPU address of the allNeurons struct on device memory.
       ///  @param  allEdgesDevice      GPU address of the allEdges struct on device memory.
       ///  @param  randNoise              Reference to the random noise array.
       ///  @param  edgeIndexMapDevice  GPU address of the EdgeIndexMap on device memory.
       virtual void advanceVertices(AllEdges &synapses, void* allVerticesDevice, void* allEdgesDevice, float* randNoise, EdgeIndexMap* edgeIndexMapDevice) override;

#else  // !defined(USE_GPU)
protected:

   ///  Helper for #advanceNeuron. Updates state of a single neuron.
   ///
   ///  @param  index Index of the neuron to update.
   virtual void advanceNeuron(const int index);

   ///  Initiates a firing of a neuron to connected neurons.
   ///
   ///  @param  index Index of the neuron to fire.
   virtual void fire(const int index) const;

#endif // defined(USE_GPU)
};

