/**
 * @file AllIFNeurons.cpp
 *
 * @brief A container of all Integate and Fire (IF) neuron data
 *
 * @ingroup Simulator/Vertices
 */

#include "AllIFNeurons.h"
#include "ParseParamError.h"
#include "Layout.h"
#include "MTRand.h"
#include "Norm.h"
#include "ParameterManager.h"
#include "OperationManager.h"

// Default constructor
AllIFNeurons::AllIFNeurons() {
   C1_ = nullptr;
   C2_ = nullptr;
   Cm_ = nullptr;
   I0_ = nullptr;
   Iinject_ = nullptr;
   Inoise_ = nullptr;
   Isyn_ = nullptr;
   Rm_ = nullptr;
   Tau_ = nullptr;
   Trefract_ = nullptr;
   Vinit_ = nullptr;
   Vm_ = nullptr;
   Vreset_ = nullptr;
   Vrest_ = nullptr;
   Vthresh_ = nullptr;
   numStepsInRefractoryPeriod_ = nullptr;
}

AllIFNeurons::~AllIFNeurons() {
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

   C1_ = nullptr;
   C2_ = nullptr;
   Cm_ = nullptr;
   I0_ = nullptr;
   Iinject_ = nullptr;
   Inoise_ = nullptr;
   Isyn_ = nullptr;
   Rm_ = nullptr;
   Tau_ = nullptr;
   Trefract_ = nullptr;
   Vinit_ = nullptr;
   Vm_ = nullptr;
   Vreset_ = nullptr;
   Vrest_ = nullptr;
   Vthresh_ = nullptr;
   numStepsInRefractoryPeriod_ = nullptr;
}

///  Setup the internal structure of the class (allocate memories).
void AllIFNeurons::setupVertices() {
   AllSpikingNeurons::setupVertices();

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

///  Load member variables from configuration file.
///  Registered to OperationManager as Operation::op::loadParameters
void AllIFNeurons::loadParameters() {
   ParameterManager::getInstance().getBGFloatByXpath("//Iinject/min/text()", IinjectRange_[0]);
   ParameterManager::getInstance().getBGFloatByXpath("//Iinject/max/text()", IinjectRange_[1]);

   ParameterManager::getInstance().getBGFloatByXpath("//Inoise/min/text()", InoiseRange_[0]);
   ParameterManager::getInstance().getBGFloatByXpath("//Inoise/max/text()", InoiseRange_[1]);

   ParameterManager::getInstance().getBGFloatByXpath("//Vthresh/min/text()", VthreshRange_[0]);
   ParameterManager::getInstance().getBGFloatByXpath("//Vthresh/max/text()", VthreshRange_[1]);

   ParameterManager::getInstance().getBGFloatByXpath("//Vresting/min/text()", VrestingRange_[0]);
   ParameterManager::getInstance().getBGFloatByXpath("//Vresting/max/text()", VrestingRange_[1]);

   ParameterManager::getInstance().getBGFloatByXpath("//Vreset/min/text()", VresetRange_[0]);
   ParameterManager::getInstance().getBGFloatByXpath("//Vreset/max/text()", VresetRange_[1]);

   ParameterManager::getInstance().getBGFloatByXpath("//Vinit/min/text()", VinitRange_[0]);
   ParameterManager::getInstance().getBGFloatByXpath("//Vinit/max/text()", VinitRange_[1]);

   ParameterManager::getInstance().getBGFloatByXpath("//starter_vthresh/min/text()", starterVthreshRange_[0]);
   ParameterManager::getInstance().getBGFloatByXpath("//starter_vthresh/max/text()", starterVthreshRange_[1]);

   ParameterManager::getInstance().getBGFloatByXpath("//starter_vreset/min/text()", starterVresetRange_[0]);
   ParameterManager::getInstance().getBGFloatByXpath("//starter_vreset/max/text()", starterVresetRange_[1]);
}

///  Prints out all parameters of the neurons to logging file.
///  Registered to OperationManager as Operation::printParameters
void AllIFNeurons::printParameters() const {
   AllVertices::printParameters();

   LOG4CPLUS_DEBUG(fileLogger_,
                   "\n\tInterval of constant injected current: ["
                         << IinjectRange_[0] << ", " << IinjectRange_[1] << "]"
                         << endl
                         << "\tInterval of STD of (gaussian) noise current: ["
                         << InoiseRange_[0] << ", " << InoiseRange_[1] << "]"
                         << endl
                         << "\tInterval of firing threshold: ["
                         << VthreshRange_[0] << ", " << VthreshRange_[1] << "]"
                         << endl
                         << "\tInterval of asymptotic voltage (Vresting): [" << VrestingRange_[0]
                         << ", " << VrestingRange_[1] << "]"
                         << endl
                         << "\tInterval of reset voltage: [" << VresetRange_[0]
                         << ", " << VresetRange_[1] << "]"
                         << endl
                         << "\tInterval of initial membrance voltage: [" << VinitRange_[0]
                         << ", " << VinitRange_[1] << "]"
                         << endl
                         << "\tStarter firing threshold: [" << starterVthreshRange_[0]
                         << ", " << starterVthreshRange_[1] << "]"
                         << endl
                         << "\tStarter reset threshold: [" << starterVresetRange_[0]
                         << ", " << starterVresetRange_[1] << "]"
                         << endl << endl);
}

///  Creates all the Neurons and generates data for them.
///
///  @param  layout      Layout information of the neural network.
void AllIFNeurons::createAllVertices(Layout *layout) {
   /* set their specific types */
   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      setNeuronDefaults(i);

      // set the neuron info for neurons
      createNeuron(i, layout);
   }
}

///  Creates a single Neuron and generates data for it.
///
///  @param  i Index of the neuron to create.
///  @param  layout       Layout information of the neural network.
void AllIFNeurons::createNeuron(int i, Layout *layout) {
   // set the neuron info for neurons
   Iinject_[i] = initRNG.inRange(IinjectRange_[0], IinjectRange_[1]);
   Inoise_[i] = initRNG.inRange(InoiseRange_[0], InoiseRange_[1]);
   Vthresh_[i] = initRNG.inRange(VthreshRange_[0], VthreshRange_[1]);
   Vrest_[i] = initRNG.inRange(VrestingRange_[0], VrestingRange_[1]);
   Vreset_[i] = initRNG.inRange(VresetRange_[0], VresetRange_[1]);
   Vinit_[i] = initRNG.inRange(VinitRange_[0], VinitRange_[1]);
   Vm_[i] = Vinit_[i];

   initNeuronConstsFromParamValues(i, Simulator::getInstance().getDeltaT());

   int maxSpikes = (int) ((Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getMaxFiringRate()));
   spikeHistory_[i] = new uint64_t[maxSpikes];
   for (int j = 0; j < maxSpikes; ++j) {
      spikeHistory_[i][j] = ULONG_MAX;
   }

   switch (layout->vertexTypeMap_[i]) {
      case INH:
         LOG4CPLUS_DEBUG(vertexLogger_, "Setting inhibitory neuron: " << i);
         // set inhibitory absolute refractory period
         Trefract_[i] = DEFAULT_InhibTrefract;// TODO(derek): move defaults inside model.
         break;

      case EXC:
         LOG4CPLUS_DEBUG(vertexLogger_, "Setting excitatory neuron: " << i);
         // set excitatory absolute refractory period
         Trefract_[i] = DEFAULT_ExcitTrefract;
         break;

      default:
         LOG4CPLUS_DEBUG(vertexLogger_, "ERROR: unknown neuron type: "
               << layout->vertexTypeMap_[i] << "@" << i);
         assert(false);
         break;
   }
   // endogenously_active_neuron_map -> Model State
   if (layout->starterMap_[i]) {
      // set endogenously active threshold voltage, reset voltage, and refractory period
      Vthresh_[i] = initRNG.inRange(starterVthreshRange_[0], starterVthreshRange_[1]);
      Vreset_[i] = initRNG.inRange(starterVresetRange_[0], starterVresetRange_[1]);
      Trefract_[i] = DEFAULT_ExcitTrefract; // TODO(derek): move defaults inside model.
   }

   LOG4CPLUS_DEBUG(vertexLogger_, "\nCREATE NEURON[" << i << "] {" << endl
                 << "\tVm = " << Vm_[i] << endl
                 << "\tVthresh = " << Vthresh_[i] << endl
                 << "\tI0 = " << I0_[i] << endl
                 << "\tInoise = " << Inoise_[i] << " from : (" << InoiseRange_[0] << "," << InoiseRange_[1]
                 << ")"
                 << endl
                 << "\tC1 = " << C1_[i] << endl
                 << "\tC2 = " << C2_[i] << endl
                 << "}" << endl);
}

///  Set the Neuron at the indexed location to default values.
///
///  @param  index    Index of the Neuron that the synapse belongs to.
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

///  Initializes the Neuron constants at the indexed location.
///
///  @param  i    Index of the Neuron.
///  @param  deltaT          Inner simulation step duration
void AllIFNeurons::initNeuronConstsFromParamValues(int i, const BGFLOAT deltaT) {
   BGFLOAT &Tau = this->Tau_[i];
   BGFLOAT &C1 = this->C1_[i];
   BGFLOAT &C2 = this->C2_[i];
   BGFLOAT &Rm = this->Rm_[i];
   BGFLOAT &I0 = this->I0_[i];
   BGFLOAT &Iinject = this->Iinject_[i];
   BGFLOAT &Vrest = this->Vrest_[i];

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

///  Outputs state of the neuron chosen as a string.
///
///  @param  index  index of the neuron (in neurons) to output info from.
///  @return the complete state of the neuron.
string AllIFNeurons::toString(const int index) const {
   stringstream ss;
   ss << "Cm: " << Cm_[index] << " "; // membrane capacitance
   ss << "Rm: " << Rm_[index] << " "; // membrane resistance
   ss << "Vthresh: " << Vthresh_[index] << " "; // if Vm exceeds, Vthresh, a spike is emitted
   ss << "Vrest: " << Vrest_[index] << " "; // the resting membrane voltage
   ss << "Vreset: " << Vreset_[index] << " "; // The voltage to reset Vm to after a spike
   ss << "Vinit: " << Vinit_[index] << endl; // The initial condition for V_m at t=0
   ss << "Trefract: " << Trefract_[index] << " "; // the number of steps in the refractory period
   ss << "Inoise: " << Inoise_[index] << " "; // the stdev of the noise to be added each delta_t
   ss << "Iinject: " << Iinject_[index] << " "; // A constant current to be injected into the LIF neuron
   ss << "nStepsInRefr: " << numStepsInRefractoryPeriod_[index]
      << endl; // the number of steps left in the refractory period
   ss << "Vm: " << Vm_[index] << " "; // the membrane voltage
   ss << "hasFired: " << hasFired_[index] << " "; // it done fired?
   ss << "C1: " << C1_[index] << " ";
   ss << "C2: " << C2_[index] << " ";
   ss << "I0: " << I0_[index] << " ";
   return ss.str();
}

///  Sets the data for Neurons to input's data.
///
///  @param  input       istream to read from.
void AllIFNeurons::deserialize(istream &input) {
   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      readNeuron(input, i);
   }
}

///  Sets the data for Neuron #index to input's data.
///
///  @param  input       istream to read from.
///  @param  i           index of the neuron (in neurons).
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

///  Writes out the data in Neurons.
///
///  @param  output      stream to write out to.
void AllIFNeurons::serialize(ostream &output) const {
   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      writeNeuron(output, i);
   }
}

///  Writes out the data in the selected Neuron.
///
///  @param  output      stream to write out to.
///  @param  i           index of the neuron (in neurons).
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


