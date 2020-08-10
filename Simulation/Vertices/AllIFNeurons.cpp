#include "AllIFNeurons.h"
#include "ParseParamError.h"
#include "Layout.h"
#include "RNG/MersenneTwister.h"
#include "RNG/Norm.h"

// Default constructor
AllIFNeurons::AllIFNeurons() : AllSpikingNeurons() {
   C1_ = NULL;
   C2_ = NULL;
   Cm_ = NULL;
   I0_ = NULL;
   Iinject_ = NULL;
   Inoise_ = NULL;
   Isyn_ = NULL;
   Rm_ = NULL;
   Tau_ = NULL;
   Trefract_ = NULL;
   Vinit_ = NULL;
   Vm_ = NULL;
   Vreset_ = NULL;
   Vrest_ = NULL;
   Vthresh_ = NULL;
   numStepsInRefractoryPeriod_ = NULL;
}

AllIFNeurons::~AllIFNeurons() {
   freeResources();
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 */
void AllIFNeurons::setupNeurons() {
   AllSpikingNeurons::setupNeurons();

   // TODO: Rename variables for easier identification
   C1_ = new BGFLOAT[size_];
   C2_ = new BGFLOAT[size_];
   Cm_ = new BGFLOAT[size_];
   I0_ = new BGFLOAT[size_];
   Iinject_ = new BGFLOAT[size_];
   Inoise_ = new BGFLOAT[size_];
   Isyn_ = new BGFLOAT[size_];
   Rm_ = new BGFLOAT[size_];
   Tau_ = new BGFLOAT[size_];
   Trefract_ = new BGFLOAT[size_];
   Vinit_ = new BGFLOAT[size_];
   Vm_ = new BGFLOAT[size_];
   Vreset_ = new BGFLOAT[size_];
   Vrest_ = new BGFLOAT[size_];
   Vthresh_ = new BGFLOAT[size_];
   numStepsInRefractoryPeriod_ = new int[size_];

   for (int i = 0; i < size_; ++i) {
      numStepsInRefractoryPeriod_[i] = 0;
   }
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllIFNeurons::cleanupNeurons() {
   freeResources();
   AllSpikingNeurons::cleanupNeurons();
}

/**
 *  Deallocate all resources
 */
void AllIFNeurons::freeResources() {
   if (size_ != 0) {
      delete[] C1_;
      delete[] C2_;
      delete[] Cm_;
      delete[] I0_;
      delete[] Iinject_;
      delete[] Inoise_;
      delete[] Isyn_;
      delete[] Rm_;
      delete[] Tau_;
      delete[] Trefract_;
      delete[] Vinit_;
      delete[] Vm_;
      delete[] Vreset_;
      delete[] Vrest_;
      delete[] Vthresh_;
      delete[] numStepsInRefractoryPeriod_;
   }

   C1_ = NULL;
   C2_ = NULL;
   Cm_ = NULL;
   I0_ = NULL;
   Iinject_ = NULL;
   Inoise_ = NULL;
   Isyn_ = NULL;
   Rm_ = NULL;
   Tau_ = NULL;
   Trefract_ = NULL;
   Vinit_ = NULL;
   Vm_ = NULL;
   Vreset_ = NULL;
   Vrest_ = NULL;
   Vthresh_ = NULL;
   numStepsInRefractoryPeriod_ = NULL;
}

/*
 *  Prints out all parameters of the neurons to ostream.
 * 
 *  @param  output  ostream to send output to.
 */
void AllIFNeurons::printParameters() const {
   cout << "Interval of constant injected current: ["
          << IinjectRange_[0] << ", " << IinjectRange_[1] << "]"
          << endl;
   cout << "Interval of STD of (gaussian) noise current: ["
          << InoiseRange_[0] << ", " << InoiseRange_[1] << "]"
          << endl;
   cout << "Interval of firing threshold: ["
          << VthreshRange_[0] << ", " << VthreshRange_[1] << "]"
          << endl;
   cout << "Interval of asymptotic voltage (Vresting): [" << VrestingRange_[0]
          << ", " << VrestingRange_[1] << "]"
          << endl;
   cout << "Interval of reset voltage: [" << VresetRange_[0]
          << ", " << VresetRange_[1] << "]"
          << endl;
   cout << "Interval of initial membrance voltage: [" << VinitRange_[0]
          << ", " << VinitRange_[1] << "]"
          << endl;
   cout << "Starter firing threshold: [" << starterVthreshRange_[0]
          << ", " << starterVthreshRange_[1] << "]"
          << endl;
   cout << "Starter reset threshold: [" << starterVresetRange_[0]
          << ", " << starterVresetRange_[1] << "]"
          << endl;
}

/*
 *  Creates all the Neurons and generates data for them.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  layout      Layout information of the neunal network.
 */
void AllIFNeurons::createAllNeurons(Layout *layout) {
   /* set their specific types */
   for (int neuron_index = 0; neuron_index < Simulator::getInstance().getTotalNeurons(); neuron_index++) {
      setNeuronDefaults(neuron_index);

      // set the neuron info for neurons
      createNeuron(neuron_index, layout);
   }
}

/*
 *  Creates a single Neuron and generates data for it.
 *
 *  @param  sim_info     SimulationInfo class to read information from.
 *  @param  neuron_index Index of the neuron to create.
 *  @param  layout       Layout information of the neunal network.
 */
void AllIFNeurons::createNeuron(int neuron_index, Layout *layout) {
   // set the neuron info for neurons
   Iinject_[neuron_index] = rng.inRange(IinjectRange_[0], IinjectRange_[1]);
   Inoise_[neuron_index] = rng.inRange(InoiseRange_[0], InoiseRange_[1]);
   Vthresh_[neuron_index] = rng.inRange(VthreshRange_[0], VthreshRange_[1]);
   Vrest_[neuron_index] = rng.inRange(VrestingRange_[0], VrestingRange_[1]);
   Vreset_[neuron_index] = rng.inRange(VresetRange_[0], VresetRange_[1]);
   Vinit_[neuron_index] = rng.inRange(VinitRange_[0], VinitRange_[1]);
   Vm_[neuron_index] = Vinit_[neuron_index];

   initNeuronConstsFromParamValues(neuron_index, Simulator::getInstance().getDeltaT());

   int max_spikes = (int) ((Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getMaxFiringRate()));
   spikeHistory_[neuron_index] = new uint64_t[max_spikes];
   for (int j = 0; j < max_spikes; ++j) {
      spikeHistory_[neuron_index][j] = ULONG_MAX;
   }

   switch (layout->neuron_type_map[neuron_index]) {
      case INH: DEBUG_MID(cout << "setting inhibitory neuron: " << neuron_index << endl;)
         // set inhibitory absolute refractory period
         Trefract_[neuron_index] = DEFAULT_InhibTrefract;// TODO(derek): move defaults inside model.
         break;

      case EXC: DEBUG_MID(cout << "setting exitory neuron: " << neuron_index << endl;)
         // set excitory absolute refractory period
         Trefract_[neuron_index] = DEFAULT_ExcitTrefract;
         break;

      default: DEBUG_MID(
            cout << "ERROR: unknown neuron type: " << layout->neuron_type_map[neuron_index] << "@" << neuron_index
                 << endl;)
         assert(false);
         break;
   }
   // endogenously_active_neuron_map -> Model State
   if (layout->starter_map[neuron_index]) {
      // set endogenously active threshold voltage, reset voltage, and refractory period
      Vthresh_[neuron_index] = rng.inRange(starterVthreshRange_[0], starterVthreshRange_[1]);
      Vreset_[neuron_index] = rng.inRange(starterVresetRange_[0], starterVresetRange_[1]);
      Trefract_[neuron_index] = DEFAULT_ExcitTrefract; // TODO(derek): move defaults inside model.
   }

   DEBUG_HI(cout << "CREATE NEURON[" << neuron_index << "] {" << endl
                 << "\tVm = " << Vm_[neuron_index] << endl
                 << "\tVthresh = " << Vthresh_[neuron_index] << endl
                 << "\tI0 = " << I0_[neuron_index] << endl
                 << "\tInoise = " << Inoise_[neuron_index] << "from : (" << InoiseRange_[0] << "," << InoiseRange_[1] << ")"
                 << endl
                 << "\tC1 = " << C1_[neuron_index] << endl
                 << "\tC2 = " << C2_[neuron_index] << endl
                 << "}" << endl;)
}

/*
 *  Set the Neuron at the indexed location to default values.
 *
 *  @param  neuron_index    Index of the Neuron that the synapse belongs to.
 */
void AllIFNeurons::setNeuronDefaults(const int index) {
   Cm_[index] = DEFAULT_Cm;
   Rm_[index] = DEFAULT_Rm;
   Vthresh_[index] = DEFAULT_Vthresh;
   Vrest_[index] = DEFAULT_Vrest;
   Vreset_[index] = DEFAULT_Vreset;
   Vinit_[index] = DEFAULT_Vreset;
   Trefract_[index] = DEFAULT_Trefract;
   Inoise_[index] = DEFAULT_Inoise;
   Iinject_[index] = DEFAULT_Iinject;
   Tau_[index] = DEFAULT_Cm * DEFAULT_Rm;
}

/*
 *  Initializes the Neuron constants at the indexed location.
 *
 *  @param  neuron_index    Index of the Neuron.
 *  @param  deltaT          Inner simulation step duration
 */
void AllIFNeurons::initNeuronConstsFromParamValues(int neuron_index, const BGFLOAT deltaT) {
   BGFLOAT &Tau = this->Tau_[neuron_index];
   BGFLOAT &C1 = this->C1_[neuron_index];
   BGFLOAT &C2 = this->C2_[neuron_index];
   BGFLOAT &Rm = this->Rm_[neuron_index];
   BGFLOAT &I0 = this->I0_[neuron_index];
   BGFLOAT &Iinject = this->Iinject_[neuron_index];
   BGFLOAT &Vrest = this->Vrest_[neuron_index];

   /* init consts C1,C2 for exponential Euler integration */
   if (Tau > 0) {
      C1 = exp(-deltaT / Tau);
      C2 = Rm * (1 - C1);
   } else {
      C1 = 0.0;
      C2 = Rm;
   }
   /* calculate const IO */
   if (Rm > 0) {
      I0 = Iinject + Vrest / Rm;
   } else {
      assert(false);
   }
}

/*
 *  Outputs state of the neuron chosen as a string.
 *
 *  @param  i   index of the neuron (in neurons) to output info from.
 *  @return the complete state of the neuron.
 */
string AllIFNeurons::toString(const int i) const {
   stringstream ss;
   ss << "Cm: " << Cm_[i] << " "; // membrane capacitance
   ss << "Rm: " << Rm_[i] << " "; // membrane resistance
   ss << "Vthresh: " << Vthresh_[i] << " "; // if Vm exceeds, Vthresh, a spike is emitted
   ss << "Vrest: " << Vrest_[i] << " "; // the resting membrane voltage
   ss << "Vreset: " << Vreset_[i] << " "; // The voltage to reset Vm to after a spike
   ss << "Vinit: " << Vinit_[i] << endl; // The initial condition for V_m at t=0
   ss << "Trefract: " << Trefract_[i] << " "; // the number of steps in the refractory period
   ss << "Inoise: " << Inoise_[i] << " "; // the stdev of the noise to be added each delta_t
   ss << "Iinject: " << Iinject_[i] << " "; // A constant current to be injected into the LIF neuron
   ss << "nStepsInRefr: " << numStepsInRefractoryPeriod_[i] << endl; // the number of steps left in the refractory period
   ss << "Vm: " << Vm_[i] << " "; // the membrane voltage
   ss << "hasFired: " << hasFired_[i] << " "; // it done fired?
   ss << "C1: " << C1_[i] << " ";
   ss << "C2: " << C2_[i] << " ";
   ss << "I0: " << I0_[i] << " ";
   return ss.str();
}

/*
 *  Sets the data for Neurons to input's data.
 *
 *  @param  input       istream to read from.
 *  @param  sim_info    used as a reference to set info for neuronss.
 */
void AllIFNeurons::deserialize(istream &input) {
   for (int i = 0; i < Simulator::getInstance().getTotalNeurons(); i++) {
      readNeuron(input, i);
   }
}

/*
 *  Sets the data for Neuron #index to input's data.
 *
 *  @param  input       istream to read from.
 *  @param  sim_info    used as a reference to set info for neurons.
 *  @param  i           index of the neuron (in neurons).
 */
void AllIFNeurons::readNeuron(istream &input, int i) {
   // input.ignore() so input skips over end-of-line characters.
   input >> Cm_[i];
   input.ignore();
   input >> Rm_[i];
   input.ignore();
   input >> Vthresh_[i];
   input.ignore();
   input >> Vrest_[i];
   input.ignore();
   input >> Vreset_[i];
   input.ignore();
   input >> Vinit_[i];
   input.ignore();
   input >> Trefract_[i];
   input.ignore();
   input >> Inoise_[i];
   input.ignore();
   input >> Iinject_[i];
   input.ignore();
   input >> Isyn_[i];
   input.ignore();
   input >> numStepsInRefractoryPeriod_[i];
   input.ignore();
   input >> C1_[i];
   input.ignore();
   input >> C2_[i];
   input.ignore();
   input >> I0_[i];
   input.ignore();
   input >> Vm_[i];
   input.ignore();
   input >> hasFired_[i];
   input.ignore();
   input >> Tau_[i];
   input.ignore();
}

/*
 *  Writes out the data in Neurons.
 *
 *  @param  output      stream to write out to.
 *  @param  sim_info    used as a reference to set info for neuronss.
 */
void AllIFNeurons::serialize(ostream &output) const {
   for (int i = 0; i < Simulator::getInstance().getTotalNeurons(); i++) {
      writeNeuron(output, i);
   }
}

/*
 *  Writes out the data in the selected Neuron.
 *
 *  @param  output      stream to write out to.
 *  @param  sim_info    used as a reference to set info for neuronss.
 *  @param  i           index of the neuron (in neurons).
 */
void AllIFNeurons::writeNeuron(ostream &output, int i) const {
   output << Cm_[i] << ends;
   output << Rm_[i] << ends;
   output << Vthresh_[i] << ends;
   output << Vrest_[i] << ends;
   output << Vreset_[i] << ends;
   output << Vinit_[i] << ends;
   output << Trefract_[i] << ends;
   output << Inoise_[i] << ends;
   output << Iinject_[i] << ends;
   output << Isyn_[i] << ends;
   output << numStepsInRefractoryPeriod_[i] << ends;
   output << C1_[i] << ends;
   output << C2_[i] << ends;
   output << I0_[i] << ends;
   output << Vm_[i] << ends;
   output << hasFired_[i] << ends;
   output << Tau_[i] << ends;
}
