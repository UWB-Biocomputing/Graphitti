/**
 * @file AllIFNeurons.h
 * 
 * @ingroup Simulator/Vertices
 *
 * @brief A container of all Integate and Fire (IF) neuron data
 * 
 * A container of all Integate and Fire (IF) neuron data.
 * This is the base class of all Integate and Fire (IF) neuron classes.
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
 */
#pragma once

#include "AllSpikingNeurons.h"
#include "DeviceVector.h"
#include "Global.h"
// cereal
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>

//AllIFNeuronsDeviceProperties is alias for AllSpikingNeuronsDeviceProperties
//for now, to avoid removing the AllIFNeuronsDeviceProperties struct, but
//eventually it can be removed from all the places where it is used and
//instead replaced by AllSpikingNeuronsDeviceProperties(base struct) itself.
#if defined(USE_GPU)
using AllIFNeuronsDeviceProperties = AllSpikingNeuronsDeviceProperties;
#endif   // defined(USE_GPU)

class AllIFNeurons : public AllSpikingNeurons {
public:
   // Default neuron parameter values
   static constexpr BGFLOAT DEFAULT_Cm = 3e-8;         // The default membrane capacitance
   static constexpr BGFLOAT DEFAULT_Rm = 1e6;          // The default membrane resistance
   static constexpr BGFLOAT DEFAULT_Vrest = 0.0;       // The default resting voltage
   static constexpr BGFLOAT DEFAULT_Vreset = -0.06;    // The default reset voltage
   static constexpr BGFLOAT DEFAULT_Trefract = 3e-3;   // The default absolute refractory period
   static constexpr BGFLOAT DEFAULT_Inoise = 0.0;      // The default synaptic noise
   static constexpr BGFLOAT DEFAULT_Iinject = 0.0;     // The default injected current
   static constexpr BGFLOAT DEFAULT_Vthresh = -0.04;   // The default threshold voltage
   static constexpr BGFLOAT DEFAULT_InhibTrefract
      = 2.0e-3;   // The default absolute refractory period for inhibitory neurons
   static constexpr BGFLOAT DEFAULT_ExcitTrefract
      = 3.0e-3;   // The default absolute refractory period for excitatory neurons

   AllIFNeurons() = default;

   virtual ~AllIFNeurons() = default;

   ///  Setup the internal structure of the class.
   ///  Allocate memories to store all neurons' state.
   virtual void setupVertices() override;

   ///  Load member variables from configuration file.
   ///  Registered to OperationManager as Operation::loadParameters
   virtual void loadParameters();

   ///  Prints out all parameters of the neurons to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const;

   ///  Creates all the Neurons and assigns initial data for them.
   ///
   ///  @param  layout      Layout information of the neural network.
   virtual void createAllVertices(Layout &layout);

   ///  Outputs state of the neuron chosen as a string.
   ///
   ///  @param  index   index of the neuron (in neurons) to output info from.
   ///  @return the complete state of the neuron.
   virtual string toString(int index) const;

   /// Reads and sets the data for all neurons from input stream.
   ///
   /// @param  input       istream to read from.
   virtual void deserialize(istream &input);

   ///  Writes out the data in all neurons to output stream.
   ///
   ///  @param  output      stream to write out to.ss.
   virtual void serialize(ostream &output) const;

   ///  Cereal serialization method
   template <class Archive> void serialize(Archive &archive);

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
   virtual void advanceVertices(AllEdges &synapses, void *allVerticesDevice, void *allEdgesDevice,
                                float randNoise[], EdgeIndexMapDevice *edgeIndexMapDevice);

   ///  Allocate GPU memories to store all neurons' states,
   ///  and copy them from host to GPU memory.
   virtual void allocVerticesDeviceStruct();

   ///  Delete GPU memories.
   ///
   virtual void deleteVerticesDeviceStruct();

   ///  Clear the spike counts out of all neurons.
   //
   ///  @param  allVerticesDevice   GPU address of the allNeurons struct on device memory.
   virtual void clearVertexHistory(void *allVerticesDevice) override;
   //Copy all neurons' data from device to host.
   virtual void copyFromDevice() override;
   //Copy all neurons' data from host to device.
   virtual void copyToDevice() override;

protected:
   ///  Allocate GPU memories to store all neurons' states.
   ///  (Helper function of allocVerticesDeviceStruct)
   ///  @param  allVerticesDevice         Reference to the AllIFNeuronsDeviceProperties struct.
   void allocDeviceStruct(AllIFNeuronsDeviceProperties &allVerticesDevice);

   ///  Delete GPU memories.
   ///  (Helper function of deleteVerticesDeviceStruct)
   ///
   ///  @param  allVerticesDevice         Reference to the AllIFNeuronsDeviceProperties struct.
   void deleteDeviceStruct(AllIFNeuronsDeviceProperties &allVerticesDevice);

   // ///  Copy all neurons' data from host to device.
   // ///  (Helper function of copyNeuronHostToDevice)
   // ///
   // ///  @param  allVerticesDevice         Reference to the AllIFNeuronsDeviceProperties struct.
   // void copyHostToDevice(AllIFNeuronsDeviceProperties &allVerticesDevice);

   ///  Copy all neurons' data from device to host.
   ///  (Helper function of copyNeuronDeviceToHost)
   ///
   ///  @param  allVerticesDevice         Reference to the AllIFNeuronsDeviceProperties struct.
   void copyDeviceToHost(AllIFNeuronsDeviceProperties &allVerticesDevice);

#endif   // defined(USE_GPU)

protected:
   ///  Creates a single Neuron and generates data for it.
   ///
   ///  @param  neuronIndex Index of the neuron to create.
   ///  @param  layout       Layout information of the neural network.
   void createNeuron(int neuronIndex, Layout &layout);

   ///  Set the Neuron at the indexed location to default values.
   ///
   ///  @param  index    Index of the Neuron that the synapse belongs to.
   void setNeuronDefaults(int index);

   ///  Initializes the Neuron constants at the indexed location.
   ///
   ///  @param  neuronIndex    Index of the Neuron.
   ///  @param  deltaT          Inner simulation step duration
   virtual void initNeuronConstsFromParamValues(int neuronIndex, BGFLOAT deltaT);

   ///  Sets the data for Neuron #index to input's data.
   ///
   ///  @param  input       istream to read from.
   ///  @param  i           index of the neuron (in neurons).
   void readNeuron(istream &input, int i);

   ///  Writes out the data in the selected Neuron.
   ///
   ///  @param  output      stream to write out to.
   ///  @param  i           index of the neuron (in neurons).
   void writeNeuron(ostream &output, int i) const;

public:
   ///  The length of the absolute refractory period. [units=sec; range=(0,1);]
   DeviceVector<BGFLOAT> Trefract_;

   ///  If \f$V_m\f$ exceeds \f$V_{thresh}\f$ a spike is emmited. [units=V; range=(-10,100);]
   DeviceVector<BGFLOAT> Vthresh_;

   ///  The resting membrane voltage. [units=V; range=(-1,1);]
   DeviceVector<BGFLOAT> Vrest_;

   ///  The voltage to reset \f$V_m\f$ to after a spike. [units=V; range=(-1,1);]
   DeviceVector<BGFLOAT> Vreset_;

   ///  The initial condition for \f$V_m\f$ at time \f$t=0\f$. [units=V; range=(-1,1);]
   DeviceVector<BGFLOAT> Vinit_;

   ///  The membrane capacitance \f$C_m\f$ [range=(0,1); units=F;]
   ///  Used to initialize Tau (no use after that)
   DeviceVector<BGFLOAT> Cm_;

   ///  The membrane resistance \f$R_m\f$ [units=Ohm; range=(0,1e30)]
   DeviceVector<BGFLOAT> Rm_;

   /// The standard deviation of the noise to be added each integration time constant. [range=(0,1); units=A;]
   DeviceVector<BGFLOAT> Inoise_;

   ///  A constant current to be injected into the LIF neuron. [units=A; range=(-1,1);]
   DeviceVector<BGFLOAT> Iinject_;

   /// What the hell is this used for???
   ///  It does not seem to be used; seems to be a candidate for deletion.
   ///  Possibly from the old code before using a separate summation point
   ///  The synaptic input current.
   DeviceVector<BGFLOAT> Isyn_;

   /// The remaining number of time steps for the absolute refractory period.
   DeviceVector<int> numStepsInRefractoryPeriod_;

   /// Internal constant for the exponential Euler integration of f$V_m\f$.
   DeviceVector<BGFLOAT> C1_;

   /// Internal constant for the exponential Euler integration of \f$V_m\f$.
   DeviceVector<BGFLOAT> C2_;

   /// Internal constant for the exponential Euler integration of \f$V_m\f$.
   DeviceVector<BGFLOAT> I0_;

   /// The membrane voltage \f$V_m\f$ [readonly; units=V;]
   DeviceVector<BGFLOAT> Vm_;

   /// The membrane time constant \f$(R_m \cdot C_m)\f$
   DeviceVector<BGFLOAT> Tau_;

private:
   /// Min/max values of Iinject.
   BGFLOAT IinjectRange_[2];

   /// Min/max values of Inoise.
   BGFLOAT InoiseRange_[2];

   /// Min/max values of Vthresh.
   BGFLOAT VthreshRange_[2];

   /// Min/max values of Vresting.
   BGFLOAT VrestingRange_[2];

   /// Min/max values of Vreset.
   BGFLOAT VresetRange_[2];

   /// Min/max values of Vinit.
   BGFLOAT VinitRange_[2];

   /// Min/max values of Vthresh.
   BGFLOAT starterVthreshRange_[2];

   /// Min/max values of Vreset.
   BGFLOAT starterVresetRange_[2];
};

CEREAL_REGISTER_TYPE(AllIFNeurons);

///  Cereal serialization method
template <class Archive> void AllIFNeurons::serialize(Archive &archive)
{
   // DeviceVector::getHostVector() returns a copy, so cereal would deserialize into a
   // temporary and never restore simulation state. Cast to std::vector& so load/save
   // targets the underlying hostData_ used by advanceNeuron() and related CPU paths.
   archive(
      cereal::base_class<AllSpikingNeurons>(this),
      cereal::make_nvp("Trefract", static_cast<std::vector<BGFLOAT> &>(Trefract_)),
      cereal::make_nvp("Vthresh", static_cast<std::vector<BGFLOAT> &>(Vthresh_)),
      cereal::make_nvp("Vrest", static_cast<std::vector<BGFLOAT> &>(Vrest_)),
      cereal::make_nvp("Vreset", static_cast<std::vector<BGFLOAT> &>(Vreset_)),
      cereal::make_nvp("Vinit", static_cast<std::vector<BGFLOAT> &>(Vinit_)),
      cereal::make_nvp("Cm", static_cast<std::vector<BGFLOAT> &>(Cm_)),
      cereal::make_nvp("Rm", static_cast<std::vector<BGFLOAT> &>(Rm_)),
      cereal::make_nvp("Inoise", static_cast<std::vector<BGFLOAT> &>(Inoise_)),
      cereal::make_nvp("Iinject", static_cast<std::vector<BGFLOAT> &>(Iinject_)),
      cereal::make_nvp("Isyn", static_cast<std::vector<BGFLOAT> &>(Isyn_)),
      cereal::make_nvp("numStepsInRefractoryPeriod",
                      static_cast<std::vector<int> &>(numStepsInRefractoryPeriod_)),
      cereal::make_nvp("C1", static_cast<std::vector<BGFLOAT> &>(C1_)),
      cereal::make_nvp("C2", static_cast<std::vector<BGFLOAT> &>(C2_)),
      cereal::make_nvp("I0", static_cast<std::vector<BGFLOAT> &>(I0_)),
      cereal::make_nvp("Vm", static_cast<std::vector<BGFLOAT> &>(Vm_)),
      cereal::make_nvp("Tau", static_cast<std::vector<BGFLOAT> &>(Tau_)));

   //Private variables are intentionally excluded from serialization as they are populated from configuration files.
}