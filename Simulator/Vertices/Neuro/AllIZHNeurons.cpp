/**
 * @file AllIZHNeurons.cpp
 *
 * @brief A container of all Izhikevich neuron data
 *
 * @ingroup Simulator/Vertices
 */

#include "AllIZHNeurons.h"
#include "ParseParamError.h"

// Default constructor
AllIZHNeurons::AllIZHNeurons() : AllIFNeurons() {
   Aconst_ = nullptr;
   Bconst_ = nullptr;
   Cconst_ = nullptr;
   Dconst_ = nullptr;
   u_ = nullptr;
   C3_ = nullptr;
}

AllIZHNeurons::~AllIZHNeurons() {
   if (size_ != 0) {
      delete[] Aconst_;
      delete[] Bconst_;
      delete[] Cconst_;
      delete[] Dconst_;
      delete[] u_;
      delete[] C3_;
   }

   Aconst_ = nullptr;
   Bconst_ = nullptr;
   Cconst_ = nullptr;
   Dconst_ = nullptr;
   u_ = nullptr;
   C3_ = nullptr;
}

///  Setup the internal structure of the class (allocate memories).
void AllIZHNeurons::setupVertices() {
   AllIFNeurons::setupVertices();

   Aconst_ = new BGFLOAT[size_];
   Bconst_ = new BGFLOAT[size_];
   Cconst_ = new BGFLOAT[size_];
   Dconst_ = new BGFLOAT[size_];
   u_ = new BGFLOAT[size_];
   C3_ = new BGFLOAT[size_];
}

///  Prints out all parameters of the neurons to logging file.
///  Registered to OperationManager as Operation::printParameters
void AllIZHNeurons::printParameters() const {
   AllIFNeurons::printParameters();

   LOG4CPLUS_DEBUG(fileLogger_, "\n\tVertices type: AllIZHNeurons" << endl
             << "\tInterval of A constant for excitatory neurons: ["
             << excAconst_[0] << ", " << excAconst_[1] << "]"
             << endl
             << "\tInterval of A constant for inhibitory neurons: ["
             << inhAconst_[0] << ", " << inhAconst_[1] << "]"
             << endl
             << "\tInterval of B constant for excitatory neurons: ["
             << excBconst_[0] << ", " << excBconst_[1] << "]"
             << endl
             << "\tInterval of B constant for inhibitory neurons: ["
             << inhBconst_[0] << ", " << inhBconst_[1] << "]"
             << endl
             << "\tInterval of C constant for excitatory neurons: ["
             << excCconst_[0] << ", " << excCconst_[1] << "]"
             << endl
             << "\tInterval of C constant for inhibitory neurons: ["
             << inhCconst_[0] << ", " << inhCconst_[1] << "]"
             << endl
             << "\tInterval of D constant for excitatory neurons: ["
             << excDconst_[0] << ", " << excDconst_[1] << "]"
             << endl
             << "\tInterval of D constant for inhibitory neurons: ["
             << inhDconst_[0] << ", " << inhDconst_[1] << "]"
             << endl << endl);

}

///  Creates all the Neurons and generates data for them.
///
///  @param  layout      Layout information of the neural network.
void AllIZHNeurons::createAllVertices(Layout *layout) {
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
void AllIZHNeurons::createNeuron(int i, Layout *layout) {
   // set the neuron info for neurons
   AllIFNeurons::createNeuron(i, layout);

   // TODO: we may need another distribution mode besides flat distribution
   if (layout->vertexTypeMap_[i] == EXC) {
      // excitatory neuron
      Aconst_[i] = initRNG.inRange(excAconst_[0], excAconst_[1]);
      Bconst_[i] = initRNG.inRange(excBconst_[0], excBconst_[1]);
      Cconst_[i] = initRNG.inRange(excCconst_[0], excCconst_[1]);
      Dconst_[i] = initRNG.inRange(excDconst_[0], excDconst_[1]);
   } else {
      // inhibitory neuron
      Aconst_[i] = initRNG.inRange(inhAconst_[0], inhAconst_[1]);
      Bconst_[i] = initRNG.inRange(inhBconst_[0], inhBconst_[1]);
      Cconst_[i] = initRNG.inRange(inhCconst_[0], inhCconst_[1]);
      Dconst_[i] = initRNG.inRange(inhDconst_[0], inhDconst_[1]);
   }

   u_[i] = 0;

   LOG4CPLUS_DEBUG(fileLogger_, "\nCREATE NEURON[" << i << "] {" << endl
                 << "\tAconst = " << Aconst_[i] << endl
                 << "\tBconst = " << Bconst_[i] << endl
                 << "\tCconst = " << Cconst_[i] << endl
                 << "\tDconst = " << Dconst_[i] << endl
                 << "\tC3 = " << C3_[i] << endl
                 << "}" << endl);

}

///  Set the Neuron at the indexed location to default values.
///
///  @param  i    Index of the Neuron to refer.
void AllIZHNeurons::setNeuronDefaults(const int index) {
   AllIFNeurons::setNeuronDefaults(index);

   // no refractory period
   Trefract_[index] = 0;

   Aconst_[index] = DEFAULT_a;
   Bconst_[index] = DEFAULT_b;
   Cconst_[index] = DEFAULT_c;
   Dconst_[index] = DEFAULT_d;
}

///  Initializes the Neuron constants at the indexed location.
///
///  @param  i    Index of the Neuron.
///  @param  deltaT          Inner simulation step duration
void AllIZHNeurons::initNeuronConstsFromParamValues(int i, const BGFLOAT deltaT) {
   AllIFNeurons::initNeuronConstsFromParamValues(i, deltaT);

   BGFLOAT &C3 = this->C3_[i];
   C3 = deltaT * 1000;
}

///  Outputs state of the neuron chosen as a string.
///
///  @param  index index of the neuron (in neurons) to output info from.
///  @return the complete state of the neuron.
string AllIZHNeurons::toString(const int index) const {
   stringstream ss;

   ss << AllIFNeurons::toString(index);

   ss << "Aconst: " << Aconst_[index] << " ";
   ss << "Bconst: " << Bconst_[index] << " ";
   ss << "Cconst: " << Cconst_[index] << " ";
   ss << "Dconst: " << Dconst_[index] << " ";
   ss << "u: " << u_[index] << " ";
   ss << "C3: " << C3_[index] << " ";
   return ss.str();
}

///  Sets the data for Neurons to input's data.
///
///  @param  input       istream to read from.
void AllIZHNeurons::deserialize(istream &input) {
   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      readNeuron(input, i);
   }
}

///  Sets the data for Neuron #index to input's data.
///
///  @param  input       istream to read from.
///  @param  index           index of the neuron (in neurons).
void AllIZHNeurons::readNeuron(istream &input, int index) {
   AllIFNeurons::readNeuron(input, index);

   input >> Aconst_[index];
   input.ignore();
   input >> Bconst_[index];
   input.ignore();
   input >> Cconst_[index];
   input.ignore();
   input >> Dconst_[index];
   input.ignore();
   input >> u_[index];
   input.ignore();
   input >> C3_[index];
   input.ignore();
}

///  Writes out the data in Neurons.
///
///  @param  output      stream to write out to.
void AllIZHNeurons::serialize(ostream &output) const {
   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      writeNeuron(output, i);
   }
}

///  Writes out the data in the selected Neuron.
///
///  @param  output      stream to write out to.
///  @param  index       index of the neuron (in neurons).
void AllIZHNeurons::writeNeuron(ostream &output, int index) const {
   AllIFNeurons::writeNeuron(output, index);

   output << Aconst_[index] << ends;
   output << Bconst_[index] << ends;
   output << Cconst_[index] << ends;
   output << Dconst_[index] << ends;
   output << u_[index] << ends;
   output << C3_[index] << ends;
}

#if !defined(USE_GPU)

///  Update internal state of the indexed Neuron (called by every simulation step).
///
///  @param  index       Index of the Neuron to update.
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
      BGFLOAT noise = (*noiseRNG)();
      // Happens really often, causes drastic slow down
      // DEBUG_MID(cout << "ADVANCE NEURON[" << index << "] :: noise = " << noise << endl;)
      summationPoint += noise * Inoise; // add noise

      BGFLOAT Vint = Vm * 1000;

      // Izhikevich model integration step
      BGFLOAT Vb = Vint + C3 * (0.04 * Vint * Vint + 5 * Vint + 140 - u);
      u = u + C3 * a * (b * Vint - u);

      Vm = Vb * 0.001 + C2 * summationPoint;  // add inputs
   }

   // Happens really often, causes drastic slow down
//   DEBUG_MID(cout << index << " " << Vm << endl;)
//   DEBUG_MID(cout << "NEURON[" << index << "] {" << endl
//                  << "\tVm = " << Vm << endl
//                  << "\ta = " << a << endl
//                  << "\tb = " << b << endl
//                  << "\tc = " << Cconst_[index] << endl
//                  << "\td = " << Dconst_[index] << endl
//                  << "\tu = " << u << endl
//                  << "\tVthresh = " << Vthresh << endl
//                  << "\tsummationPoint = " << summationPoint << endl
//                  << "\tI0 = " << I0 << endl
//                  << "\tInoise = " << Inoise << endl
//                  << "\tC1 = " << C1 << endl
//                  << "\tC2 = " << C2 << endl
//                  << "\tC3 = " << C3 << endl
//                  << "}" << endl;)

   // clear synaptic input for next time step
   summationPoint = 0;
}

///  Fire the selected Neuron and calculate the result.
///
///  @param  index       Index of the Neuron to update.
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
