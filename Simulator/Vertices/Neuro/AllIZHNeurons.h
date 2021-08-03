/**
 * @file AllIZHNeurons.h
 * 
 * @ingroup Simulator/Vertices
 *
 * @brief A container of all Izhikevich neuron data
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
 * The Izhikevich neuron model uses the quadratic integrate-and-fire model 
 * for ordinary differential equations of the form:
 * \f[
 *  \frac{d v}{dt} = 0.04v^2 + 5v + 140 - u + (I_{syn}(t) + I_{inject} + I_{noise})
 * \f]
 * \f[
 *  \frac{d u}{dt} = a \cdot (bv - u)
 * \f]
 * with the auxiliary after-spike resetting: if \f$v\ge30\f$ mv, then \f$v=c,u=u+d\f$.
 *
 * where \f$v\f$ and \f$u\f$ are dimensionless variable, and \f$a,b,c\f$, and \f$d\f$ are dimensioless parameters. 
 * The variable \f$v\f$ represents the membrane potential of the neuron and \f$u\f$ represents a membrane 
 * recovery variable, which accounts for the activation of \f$K^+\f$ ionic currents and 
 * inactivation of \f$Na^+\f$ ionic currents, and it provides negative feedback to \f$v\f$. 
 * \f$I_{syn}(t)\f$ is the current supplied by the synapses, \f$I_{inject}\f$ is a non-specific 
 * background current and Inoise is a Gaussian random variable with zero mean and 
 * a given variance noise (Izhikevich. 2003).
 *
 * The simple Euler method combined with the exponential Euler method is used for 
 * numerical integration. 
 * 
 * One step of the simple Euler method from \f$y(t)\f$ to \f$y(t + \Delta t)\f$ is:
 *  \f$y(t + \Delta t) = y(t) + \Delta t \cdot y(t)\f$
 *
 * The main idea behind the exponential Euler rule is 
 * that many biological processes are governed by an exponential decay function. 
 * For an equation of the form:
 * \f[
 *  \frac{d y}{dt} = A - By
 * \f]
 * its scheme is given by:
 * \f[
 *  y(t+\Delta t) = y(t) \cdot \mathrm{e}^{-B \Delta t} + \frac{A}{B} \cdot (1 - \mathrm{e}^{-B \Delta t}) 
 * \f]
 * After appropriate substituting all variables, we obtain the Euler step:
 * \f[
 *  v(t+\Delta t)=v(t)+ C3 \cdot (0.04v(t)^2+5v(t)+140-u(t))+C2 \cdot (I_{syn}(t)+I_{inject}+I_{noise}+\frac{V_{resting}}{R_{m}})
 * \f]
 * \f[
 *  u(t+ \Delta t)=u(t) + C3 \cdot a \cdot (bv(t)-u(t))
 * \f]
 * where \f$\tau_{m}=C_{m} \cdot R_{m}\f$ is the membrane time constant, \f$R_{m}\f$ is the membrane resistance,
 * \f$C2 = R_m \cdot (1 - \mathrm{e}^{-\frac{\Delta t}{\tau_m}})\f$,
 * and \f$C3 = \Delta t\f$.
 *
 * Because the time scale \f$t\f$ of the Izhikevich model is \f$ms\f$ scale, 
 *  so \f$C3\f$ is : \f$C3 = \Delta t = 1000 \cdot deltaT\f$ (\f$deltaT\f$ is the simulation time step in second) \f$ms\f$.
 */
#pragma once

#include "Global.h"
#include "AllIFNeurons.h"

struct AllIZHNeuronsDeviceProperties;

// Class to hold all data necessary for all the Neurons.
class AllIZHNeurons : public AllIFNeurons {
public:
   AllIZHNeurons();

   virtual ~AllIZHNeurons();

   static AllVertices *Create() { return new AllIZHNeurons(); }

   ///  Setup the internal structure of the class.
   ///  Allocate memories to store all neurons' state.
   virtual void setupVertices() override;

   ///  Prints out all parameters of the neurons to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const override;

   ///  Creates all the Neurons and assigns initial data for them.
   ///
   ///  @param  layout      Layout information of the neural network.
   virtual void createAllVertices(Layout *layout) override;

   ///  Outputs state of the neuron chosen as a string.
   ///
   ///  @param  index   index of the neuron (in neurons) to output info from.
   ///  @return the complete state of the neuron.
   virtual string toString(const int index) const override;

   ///  Reads and sets the data for all neurons from input stream.
   ///
   ///  @param  input       istream to read from.
   virtual void deserialize(istream &input) override;

   ///  Writes out the data in all neurons to output stream.
   ///
   ///  @param  output      stream to write out to.
   virtual void serialize(ostream &output) const override;

#if defined(USE_GPU)
   public:
       ///  Update the state of all neurons for a time step
       ///  Notify outgoing synapses if neuron has fired.
       ///
       ///  @param  synapses               Reference to the allEdges struct on host memory.
       ///  @param  allVerticesDevice       Reference to the allNeurons struct on device memory.
       ///  @param  allEdgesDevice      Reference to the allEdges struct on device memory.
       ///  @param  randNoise              Reference to the random noise array.
       ///  @param  edgeIndexMapDevice  Reference to the EdgeIndexMap on device memory.
       virtual void advanceVertices(AllEdges &synapses, void* allVerticesDevice, void* allEdgesDevice, float* randNoise, EdgeIndexMap* edgeIndexMapDevice) override;

       ///  Allocate GPU memories to store all neurons' states,
       ///  and copy them from host to GPU memory.
       ///
       ///  @param  allVerticesDevice   GPU address of the allNeurons struct on device memory.
       virtual void allocNeuronDeviceStruct( void** allVerticesDevice) override;

       ///  Delete GPU memories.
       ///
       ///  @param  allVerticesDevice   GPU address of the allNeurons struct on device memory.
       virtual void deleteNeuronDeviceStruct( void* allVerticesDevice) override;

       ///  Copy all neurons' data from host to device.
       ///
       ///  @param  allVerticesDevice   GPU address of the allNeurons struct on device memory.
       virtual void copyNeuronHostToDevice( void* allVerticesDevice) override;

       ///  Copy all neurons' data from device to host.
       ///
       ///  @param  allVerticesDevice   GPU address of the allNeurons struct on device memory.
       virtual void copyNeuronDeviceToHost(void* allVerticesDevice) override;

       ///  Copy spike history data stored in device memory to host.
       ///
       ///  @param  allVerticesDevice   GPU address of the allNeurons struct on device memory.
       virtual void copyNeuronDeviceSpikeHistoryToHost( void* allVerticesDevice) override;

       ///  Copy spike counts data stored in device memory to host.
       ///
       ///  @param  allVerticesDevice   GPU address of the allNeurons struct on device memory.
       virtual void copyNeuronDeviceSpikeCountsToHost( void* allVerticesDevice) override;

       ///  Clear the spike counts out of all neurons.
       ///
       ///  @param  allVerticesDevice   GPU address of the allNeurons struct on device memory.
       virtual void clearNeuronSpikeCounts( void* allVerticesDevice) override;


   protected:
       ///  Allocate GPU memories to store all neurons' states.
       ///  (Helper function of allocNeuronDeviceStruct)
       ///
       ///  @param  allVerticesDevice         Reference to the AllIZHNeuronsDeviceProperties struct.
       void allocDeviceStruct( AllIZHNeuronsDeviceProperties &allVerticesDevice);

       ///  Delete GPU memories.
       ///  (Helper function of deleteNeuronDeviceStruct)
       ///
       ///  @param  allVerticesDevice         Reference to the AllIZHNeuronsDeviceProperties struct.
       void deleteDeviceStruct( AllIZHNeuronsDeviceProperties& allVerticesDevice);

       ///  Copy all neurons' data from host to device.
       ///  (Helper function of copyNeuronHostToDevice)
       ///
       ///  @param  allVerticesDevice         Reference to the AllIZHNeuronsDeviceProperties struct.
       void copyHostToDevice( AllIZHNeuronsDeviceProperties& allVerticesDevice);

       ///  Copy all neurons' data from device to host.
       ///  (Helper function of copyNeuronDeviceToHost)
       ///
       ///  @param  allVerticesDevice         Reference to the AllIZHNeuronsDeviceProperties struct.
       void copyDeviceToHost( AllIZHNeuronsDeviceProperties& allVerticesDevice);

#else  // !defined(USE_GPU)

protected:
   ///  Helper for #advanceNeuron. Updates state of a single neuron.
   ///
   ///  @param  index            Index of the neuron to update.
   virtual void advanceNeuron(const int index);

   ///  Initiates a firing of a neuron to connected neurons.
   ///
   ///  @param  index            Index of the neuron to fire.
   virtual void fire(const int index) const;

#endif  // defined(USE_GPU)

protected:
   ///  Creates a single Neuron and generates data for it.
   ///
   ///  @param  neuronIndex Index of the neuron to create.
   ///  @param  layout       Layout information of the neural network.
   void createNeuron(int neuronIndex, Layout *layout);

   ///  Set the Neuron at the indexed location to default values.
   ///
   ///  @param  index    Index of the Neuron that the synapse belongs to.
   void setNeuronDefaults(const int index);

   ///  Initializes the Neuron constants at the indexed location.
   ///
   ///  @param  neuronIndex    Index of the Neuron.
   ///  @param  deltaT          Inner simulation step duration
   virtual void initNeuronConstsFromParamValues(int neuronIndex, const BGFLOAT deltaT) override;

   ///  Sets the data for Neuron #index to input's data.
   ///
   ///  @param  input       istream to read from.
   ///  @param  index           index of the neuron (in neurons).
   void readNeuron(istream &input, int index);

   ///  Writes out the data in the selected Neuron.
   ///
   ///  @param  output      stream to write out to.
   ///  @param  index           index of the neuron (in neurons).
   void writeNeuron(ostream &output, int index) const;

public:
   ///  A constant (0.02, 01) describing the coupling of variable u to Vm.
   BGFLOAT *Aconst_;

   ///  A constant controlling sensitivity of u.
   BGFLOAT *Bconst_;

   ///  A constant controlling reset of Vm.
   BGFLOAT *Cconst_;

   ///  A constant controlling reset of u.
   BGFLOAT *Dconst_;

   ///  internal variable.
   BGFLOAT *u_;

   ///  Internal constant for the exponential Euler integration.
   BGFLOAT *C3_;

private:
   ///  Default value of Aconst.
   static constexpr BGFLOAT DEFAULT_a = 0.0035;

   ///  Default value of Bconst.
   static constexpr BGFLOAT DEFAULT_b = 0.2;

   ///  Default value of Cconst.
   static constexpr BGFLOAT DEFAULT_c = -50;

   ///  Default value of Dconst.
   static constexpr BGFLOAT DEFAULT_d = 2;

   ///  Min/max values of Aconst for excitatory neurons.
   BGFLOAT excAconst_[2];

   ///  Min/max values of Aconst for inhibitory neurons.
   BGFLOAT inhAconst_[2];

   ///  Min/max values of Bconst for excitatory neurons.
   BGFLOAT excBconst_[2];

   ///  Min/max values of Bconst for inhibitory neurons.
   BGFLOAT inhBconst_[2];

   ///  Min/max values of Cconst for excitatory neurons.
   BGFLOAT excCconst_[2];

   ///  Min/max values of Cconst for inhibitory neurons.
   BGFLOAT inhCconst_[2];

   ///  Min/max values of Dconst for excitatory neurons.
   BGFLOAT excDconst_[2];

   ///  Min/max values of Dconst for inhibitory neurons.
   BGFLOAT inhDconst_[2];
};

#if defined(USE_GPU)
struct AllIZHNeuronsDeviceProperties : public AllIFNeuronsDeviceProperties
{
        ///  A constant (0.02, 01) describing the coupling of variable u to Vm.
        BGFLOAT *Aconst_;

        ///  A constant controlling sensitivity of u.
        BGFLOAT *Bconst_;

        ///  A constant controlling reset of Vm. 
        BGFLOAT *Cconst_;

        ///  A constant controlling reset of u.
        BGFLOAT *Dconst_;

        ///  internal variable.
        BGFLOAT *u_;

        ///  Internal constant for the exponential Euler integration.
        BGFLOAT *C3_;
};
#endif // defined(USE_GPU)
