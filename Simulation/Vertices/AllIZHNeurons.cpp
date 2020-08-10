/*
 * AllIZHNeurons.cpp
 *
 */

#include "AllIZHNeurons.h"
#include "ParseParamError.h"

// Default constructor
AllIZHNeurons::AllIZHNeurons() : AllIFNeurons() {
   Aconst_ = NULL;
   Bconst_ = NULL;
   Cconst_ = NULL;
   Dconst_ = NULL;
   u_ = NULL;
   C3_ = NULL;
}

AllIZHNeurons::~AllIZHNeurons() {
   freeResources();
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 */
void AllIZHNeurons::setupNeurons() {
   AllIFNeurons::setupNeurons();

   Aconst_ = new BGFLOAT[size_];
   Bconst_ = new BGFLOAT[size_];
   Cconst_ = new BGFLOAT[size_];
   Dconst_ = new BGFLOAT[size_];
   u_ = new BGFLOAT[size_];
   C3_ = new BGFLOAT[size_];
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllIZHNeurons::cleanupNeurons() {
   freeResources();
   AllIFNeurons::cleanupNeurons();
}

/*
 *  Deallocate all resources
 */
void AllIZHNeurons::freeResources() {
   if (size_ != 0) {
      delete[] Aconst_;
      delete[] Bconst_;
      delete[] Cconst_;
      delete[] Dconst_;
      delete[] u_;
      delete[] C3_;
   }

   Aconst_ = NULL;
   Bconst_ = NULL;
   Cconst_ = NULL;
   Dconst_ = NULL;
   u_ = NULL;
   C3_ = NULL;
}

/*
 *  Prints out all parameters of the neurons to ostream.
 *
 *  @param  output  ostream to send output to.
 */
void AllIZHNeurons::printParameters() const {
   AllIFNeurons::printParameters();

   cout << "Interval of A constant for excitatory neurons: ["
          << excAconst_[0] << ", " << excAconst_[1] << "]"
          << endl;
   cout << "Interval of A constant for inhibitory neurons: ["
          << inhAconst_[0] << ", " << inhAconst_[1] << "]"
          << endl;
   cout << "Interval of B constant for excitatory neurons: ["
          << excBconst_[0] << ", " << excBconst_[1] << "]"
          << endl;
   cout << "Interval of B constant for inhibitory neurons: ["
          << inhBconst_[0] << ", " << inhBconst_[1] << "]"
          << endl;
   cout << "Interval of C constant for excitatory neurons: ["
          << excCconst_[0] << ", " << excCconst_[1] << "]"
          << endl;
   cout << "Interval of C constant for inhibitory neurons: ["
          << inhCconst_[0] << ", " << inhCconst_[1] << "]"
          << endl;
   cout << "Interval of D constant for excitatory neurons: ["
          << excDconst_[0] << ", " << excDconst_[1] << "]"
          << endl;
   cout << "Interval of D constant for inhibitory neurons: ["
          << inhDconst_[0] << ", " << inhDconst_[1] << "]"
          << endl;
}

/*
 *  Creates all the Neurons and generates data for them.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  layout      Layout information of the neunal network.
 */
void AllIZHNeurons::createAllNeurons(Layout *layout) {
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
void AllIZHNeurons::createNeuron(int neuron_index, Layout *layout) {
   // set the neuron info for neurons
   AllIFNeurons::createNeuron(neuron_index, layout);

   // TODO: we may need another distribution mode besides flat distribution
   if (layout->neuronTypeMap_[neuron_index] == EXC) {
      // excitatory neuron
      Aconst_[neuron_index] = rng.inRange(excAconst_[0], excAconst_[1]);
      Bconst_[neuron_index] = rng.inRange(excBconst_[0], excBconst_[1]);
      Cconst_[neuron_index] = rng.inRange(excCconst_[0], excCconst_[1]);
      Dconst_[neuron_index] = rng.inRange(excDconst_[0], excDconst_[1]);
   } else {
      // inhibitory neuron
      Aconst_[neuron_index] = rng.inRange(inhAconst_[0], inhAconst_[1]);
      Bconst_[neuron_index] = rng.inRange(inhBconst_[0], inhBconst_[1]);
      Cconst_[neuron_index] = rng.inRange(inhCconst_[0], inhCconst_[1]);
      Dconst_[neuron_index] = rng.inRange(inhDconst_[0], inhDconst_[1]);
   }

   u_[neuron_index] = 0;

   DEBUG_HI(cout << "CREATE NEURON[" << neuron_index << "] {" << endl
                 << "\tAconst = " << Aconst_[neuron_index] << endl
                 << "\tBconst = " << Bconst_[neuron_index] << endl
                 << "\tCconst = " << Cconst_[neuron_index] << endl
                 << "\tDconst = " << Dconst_[neuron_index] << endl
                 << "\tC3 = " << C3_[neuron_index] << endl
                 << "}" << endl;)

}

/*
 *  Set the Neuron at the indexed location to default values.
 *
 *  @param  neuron_index    Index of the Neuron to refer.
 */
void AllIZHNeurons::setNeuronDefaults(const int index) {
   AllIFNeurons::setNeuronDefaults(index);

   // no refractory period
   Trefract_[index] = 0;

   Aconst_[index] = DEFAULT_a;
   Bconst_[index] = DEFAULT_b;
   Cconst_[index] = DEFAULT_c;
   Dconst_[index] = DEFAULT_d;
}

/*
 *  Initializes the Neuron constants at the indexed location.
 *
 *  @param  neuron_index    Index of the Neuron.
 *  @param  deltaT          Inner simulation step duration
 */
void AllIZHNeurons::initNeuronConstsFromParamValues(int neuron_index, const BGFLOAT deltaT) {
   AllIFNeurons::initNeuronConstsFromParamValues(neuron_index, deltaT);

   BGFLOAT &C3 = this->C3_[neuron_index];
   C3 = deltaT * 1000;
}

/*
 *  Outputs state of the neuron chosen as a string.
 *
 *  @param  i   index of the neuron (in neurons) to output info from.
 *  @return the complete state of the neuron.
 */
string AllIZHNeurons::toString(const int i) const {
   stringstream ss;

   ss << AllIFNeurons::toString(i);

   ss << "Aconst: " << Aconst_[i] << " ";
   ss << "Bconst: " << Bconst_[i] << " ";
   ss << "Cconst: " << Cconst_[i] << " ";
   ss << "Dconst: " << Dconst_[i] << " ";
   ss << "u: " << u_[i] << " ";
   ss << "C3: " << C3_[i] << " ";
   return ss.str();
}

/*
 *  Sets the data for Neurons to input's data.
 *
 *  @param  input       istream to read from.
 *  @param  sim_info    used as a reference to set info for neurons.
 */
void AllIZHNeurons::deserialize(istream &input) {
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
void AllIZHNeurons::readNeuron(istream &input, int i) {
   AllIFNeurons::readNeuron(input, i);

   input >> Aconst_[i];
   input.ignore();
   input >> Bconst_[i];
   input.ignore();
   input >> Cconst_[i];
   input.ignore();
   input >> Dconst_[i];
   input.ignore();
   input >> u_[i];
   input.ignore();
   input >> C3_[i];
   input.ignore();
}

/*
 *  Writes out the data in Neurons.
 *
 *  @param  output      stream to write out to.
 *  @param  sim_info    used as a reference to set info for neuronss.
 */
void AllIZHNeurons::serialize(ostream &output) const {
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
void AllIZHNeurons::writeNeuron(ostream &output, int i) const {
   AllIFNeurons::writeNeuron(output, i);

   output << Aconst_[i] << ends;
   output << Bconst_[i] << ends;
   output << Cconst_[i] << ends;
   output << Dconst_[i] << ends;
   output << u_[i] << ends;
   output << C3_[i] << ends;
}

#if !defined(USE_GPU)

/*
 *  Update internal state of the indexed Neuron (called by every simulation step).
 *
 *  @param  index       Index of the Neuron to update.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void AllIZHNeurons::advanceNeuron(const int index) {
   BGFLOAT &Vm = this->Vm_[index];
   BGFLOAT &Vthresh = this->Vthresh_[index];
   BGFLOAT &summationPoint = this->summationMap_[index];
   BGFLOAT &I0 = this->I0_[index];
   BGFLOAT &Inoise = this->Inoise_[index];
   BGFLOAT &C1 = this->C1_[index];
   BGFLOAT &C2 = this->C2_[index];
   BGFLOAT &C3 = this->C3_[index];
   int &nStepsInRefr = this->numStepsInRefractoryPeriod_[index];

   BGFLOAT &a = Aconst_[index];
   BGFLOAT &b = Bconst_[index];
   BGFLOAT &u = this->u_[index];

   if (nStepsInRefr > 0) {
      // is neuron refractory?
      --nStepsInRefr;
   } else if (Vm >= Vthresh) {
      // should it fire?
      fire(index);
   } else {
      summationPoint += I0; // add IO
      // add noise
      BGFLOAT noise = (*rgNormrnd[0])();
      DEBUG_MID(cout << "ADVANCE NEURON[" << index << "] :: noise = " << noise << endl;)
      summationPoint += noise * Inoise; // add noise

      BGFLOAT Vint = Vm * 1000;

      // Izhikevich model integration step
      BGFLOAT Vb = Vint + C3 * (0.04 * Vint * Vint + 5 * Vint + 140 - u);
      u = u + C3 * a * (b * Vint - u);

      Vm = Vb * 0.001 + C2 * summationPoint;  // add inputs
   }

   DEBUG_MID(cout << index << " " << Vm << endl;)
   DEBUG_MID(cout << "NEURON[" << index << "] {" << endl
                  << "\tVm = " << Vm << endl
                  << "\ta = " << a << endl
                  << "\tb = " << b << endl
                  << "\tc = " << Cconst_[index] << endl
                  << "\td = " << Dconst_[index] << endl
                  << "\tu = " << u << endl
                  << "\tVthresh = " << Vthresh << endl
                  << "\tsummationPoint = " << summationPoint << endl
                  << "\tI0 = " << I0 << endl
                  << "\tInoise = " << Inoise << endl
                  << "\tC1 = " << C1 << endl
                  << "\tC2 = " << C2 << endl
                  << "\tC3 = " << C3 << endl
                  << "}" << endl;)

   // clear synaptic input for next time step
   summationPoint = 0;
}

/*
 *  Fire the selected Neuron and calculate the result.
 *
 *  @param  index       Index of the Neuron to update.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void AllIZHNeurons::fire(const int index) const {
   const BGFLOAT deltaT = Simulator::getInstance().getDeltaT();
   AllSpikingNeurons::fire(index);

   // calculate the number of steps in the absolute refractory period
   BGFLOAT &Vm = this->Vm_[index];
   int &nStepsInRefr = this->numStepsInRefractoryPeriod_[index];
   BGFLOAT &Trefract = this->Trefract_[index];

   BGFLOAT &c = Cconst_[index];
   BGFLOAT &d = Dconst_[index];
   BGFLOAT &u = this->u_[index];

   nStepsInRefr = static_cast<int> ( Trefract / deltaT + 0.5 );

   // reset to 'Vreset'
   Vm = c * 0.001;
   u = u + d;
}

#endif
