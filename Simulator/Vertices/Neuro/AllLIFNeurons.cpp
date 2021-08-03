/* @file AllLIFNeurons.cpp
 * 
 * @ingroup Simulator/Vertices
 *
 * @brief A container of all LIF neuron data
 */

#include "AllLIFNeurons.h"
#include "ParseParamError.h"

// Default constructor
AllLIFNeurons::AllLIFNeurons() : AllIFNeurons() {
}

AllLIFNeurons::~AllLIFNeurons() {
}

///  Prints out all parameters of the neurons to logging file.
///  Registered to OperationManager as Operation::printParameters
void AllLIFNeurons::printParameters() const {
   AllIFNeurons::printParameters();
   LOG4CPLUS_DEBUG(fileLogger_, "\n\tVertices Type: AllLIFNeurons" << endl);
}

#if !defined(USE_GPU)

///  Update internal state of the indexed Neuron (called by every simulation step).
///
///  @param  index       Index of the Neuron to update.
void AllLIFNeurons::advanceNeuron(const int index) {
    BGFLOAT &Vm = this->Vm_[index];
    BGFLOAT &Vthresh = this->Vthresh_[index];
    BGFLOAT &summationPoint = this->summationMap_[index];
    BGFLOAT &I0 = this->I0_[index];
    BGFLOAT &Inoise = this->Inoise_[index];
    BGFLOAT &C1 = this->C1_[index];
    BGFLOAT &C2 = this->C2_[index];
    int &nStepsInRefr = this->numStepsInRefractoryPeriod_[index];

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
        //LOG4CPLUS_DEBUG(vertexLogger_, "ADVANCE NEURON[" << index << "] :: Noise = " << noise);
        summationPoint += noise * Inoise; // add noise
        Vm = C1 * Vm + C2 * summationPoint; // decay Vm and add inputs
    }
    // clear synaptic input for next time step
    summationPoint = 0;

    // Causes a huge slowdown since it's printed so frequently
//    LOG4CPLUS_DEBUG(vertexLogger_, "Index: " << index << " Vm: " << Vm);
//    LOG4CPLUS_DEBUG(vertexLogger_, "NEURON[" << index << "] {" << endl
//                   << "\tVm = " << Vm << endl
//                   << "\tVthresh = " << Vthresh << endl
//                   << "\tsummationPoint = " << summationPoint << endl
//                   << "\tI0 = " << I0 << endl
//                   << "\tInoise = " << Inoise << endl
//                   << "\tC1 = " << C1 << endl
//                   << "\tC2 = " << C2 << endl
//                   << "}" << endl);
}

///  Fire the selected Neuron and calculate the result.
///
///  @param  index       Index of the Neuron to update.
void AllLIFNeurons::fire(const int index) const {
    const BGFLOAT deltaT = Simulator::getInstance().getDeltaT();
    AllSpikingNeurons::fire(index);

    // calculate the number of steps in the absolute refractory period
    numStepsInRefractoryPeriod_[index] = static_cast<int> ( Trefract_[index] / deltaT + 0.5 );

    // reset to 'Vreset'
    Vm_[index] = Vreset_[index];
}

#endif
