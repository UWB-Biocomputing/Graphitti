#include "AllDynamicSTDPSynapses.h"

AllDynamicSTDPSynapses::AllDynamicSTDPSynapses() : AllSTDPSynapses() {
    lastSpike_ = NULL;
    r_ = NULL;
    u_ = NULL;
    D_ = NULL;
    U_ = NULL;
    F_ = NULL;
}

AllDynamicSTDPSynapses::AllDynamicSTDPSynapses(const int num_neurons, const int max_synapses) :
      AllSTDPSynapses(num_neurons, max_synapses) {
    setupSynapses(num_neurons, max_synapses);
}

AllDynamicSTDPSynapses::~AllDynamicSTDPSynapses() {
    cleanupSynapses();
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 */
void AllDynamicSTDPSynapses::setupSynapses() {
    setupSynapses(Simulator::getInstance().getDeltaT(), Simulator::getInstance().getMaxSynapsesPerNeuron());
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 * 
 *  @param  num_neurons   Total number of neurons in the network.
 *  @param  max_synapses  Maximum number of synapses per neuron.
 */
void AllDynamicSTDPSynapses::setupSynapses(const int num_neurons, const int max_synapses) {
    AllSTDPSynapses::setupSynapses(num_neurons, max_synapses);

    BGSIZE max_total_synapses = max_synapses * num_neurons;

    if (max_total_synapses != 0) {
        lastSpike_ = new uint64_t[max_total_synapses];
        r_ = new BGFLOAT[max_total_synapses];
        u_ = new BGFLOAT[max_total_synapses];
        D_ = new BGFLOAT[max_total_synapses];
        U_ = new BGFLOAT[max_total_synapses];
        F_ = new BGFLOAT[max_total_synapses];
    }
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllDynamicSTDPSynapses::cleanupSynapses() {
    BGSIZE max_total_synapses = maxSynapsesPerNeuron_ * countNeurons_;

    if (max_total_synapses != 0) {
        delete[] lastSpike_;
        delete[] r_;
        delete[] u_;
        delete[] D_;
        delete[] U_;
        delete[] F_;
    }

    lastSpike_ = NULL;
    r_ = NULL;
    u_ = NULL;
    D_ = NULL;
    U_ = NULL;
    F_ = NULL;

    AllSTDPSynapses::cleanupSynapses();
}

/*
 *  Prints out all parameters of the neurons to console.
 */
void AllDynamicSTDPSynapses::printParameters() const {
   AllSTDPSynapses::printParameters();

   cout << "\t*AllDynamicSTDPSynapses Parameters*" << endl;
   cout << "\tEdges type: AllDynamicSTDPSynapses" << endl << endl;
}

/*
 *  Sets the data for Synapse to input's data.
 *
 *  @param  input  istream to read from.
 *  @param  iSyn   Index of the synapse to set.
 */
void AllDynamicSTDPSynapses::readSynapse(istream &input, const BGSIZE iSyn) {
    AllSTDPSynapses::readSynapse(input, iSyn);

    // input.ignore() so input skips over end-of-line characters.
    input >> lastSpike_[iSyn];
    input.ignore();
    input >> r_[iSyn];
    input.ignore();
    input >> u_[iSyn];
    input.ignore();
    input >> D_[iSyn];
    input.ignore();
    input >> U_[iSyn];
    input.ignore();
    input >> F_[iSyn];
    input.ignore();
}

/*
 *  Write the synapse data to the stream.
 *
 *  @param  output  stream to print out to.
 *  @param  iSyn    Index of the synapse to print out.
 */
void AllDynamicSTDPSynapses::writeSynapse(ostream &output, const BGSIZE iSyn) const {
    AllSTDPSynapses::writeSynapse(output, iSyn);

    output << lastSpike_[iSyn] << ends;
    output << r_[iSyn] << ends;
    output << u_[iSyn] << ends;
    output << D_[iSyn] << ends;
    output << U_[iSyn] << ends;
    output << F_[iSyn] << ends;
}

/*
 *  Reset time varying state vars and recompute decay.
 *
 *  @param  iSyn            Index of the synapse to set.
 *  @param  deltaT          Inner simulation step duration
 */
void AllDynamicSTDPSynapses::resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT) {
    AllSTDPSynapses::resetSynapse(iSyn, deltaT);

    u_[iSyn] = DEFAULT_U;
    r_[iSyn] = 1.0;
    lastSpike_[iSyn] = ULONG_MAX;
}

/*
 *  Create a Synapse and connect it to the model.
 *
 *  @param  synapses    The synapse list to reference.
 *  @param  iSyn        Index of the synapse to set.
 *  @param  source      Coordinates of the source Neuron.
 *  @param  dest        Coordinates of the destination Neuron.
 *  @param  sum_point   Summation point address.
 *  @param  deltaT      Inner simulation step duration.
 *  @param  type        Type of the Synapse to create.
 */
void AllDynamicSTDPSynapses::createSynapse(const BGSIZE iSyn, int source_index, int dest_index, BGFLOAT *sum_point,
                                           const BGFLOAT deltaT, synapseType type) {
    AllSTDPSynapses::createSynapse(iSyn, source_index, dest_index, sum_point, deltaT, type);

    U_[iSyn] = DEFAULT_U;

    BGFLOAT U;
    BGFLOAT D;
    BGFLOAT F;
    switch (type) {
        case II:
            U = 0.32;
            D = 0.144;
            F = 0.06;
            break;
        case IE:
            U = 0.25;
            D = 0.7;
            F = 0.02;
            break;
        case EI:
            U = 0.05;
            D = 0.125;
            F = 1.2;
            break;
        case EE:
            U = 0.5;
            D = 1.1;
            F = 0.05;
            break;
        default:
            assert(false);
            break;
    }

    this->U_[iSyn] = U;
    this->D_[iSyn] = D;
    this->F_[iSyn] = F;
}

#if !defined(USE_GPU)

/*
 *  Calculate the post synapse response after a spike.
 *
 *  @param  iSyn        Index of the synapse to set.
 *  @param  deltaT      Inner simulation step duration.
 */
void AllDynamicSTDPSynapses::changePSR(const BGSIZE iSyn, const BGFLOAT deltaT) {
    BGFLOAT &psr = this->psr_[iSyn];
    BGFLOAT &W = this->W_[iSyn];
    BGFLOAT &decay = this->decay_[iSyn];
    uint64_t &lastSpike = this->lastSpike_[iSyn];
    BGFLOAT &r = this->r_[iSyn];
    BGFLOAT &u = this->u_[iSyn];
    BGFLOAT &D = this->D_[iSyn];
    BGFLOAT &F = this->F_[iSyn];
    BGFLOAT &U = this->U_[iSyn];

    // adjust synapse parameters
    if (lastSpike != ULONG_MAX) {
        BGFLOAT isi = (g_simulationStep - lastSpike) * deltaT;
        r = 1 + (r * (1 - u) - 1) * exp(-isi / D);
        u = U + u * (1 - U) * exp(-isi / F);
    }
    psr += ((W / decay) * u * r);    // calculate psr
    lastSpike = g_simulationStep;        // record the time of the spike
}

#endif // !defined(USE_GPU)


/*
 *  Prints SynapsesProps data.
 */
void AllDynamicSTDPSynapses::printSynapsesProps() const {
    AllSTDPSynapses::printSynapsesProps();
    for (int i = 0; i < maxSynapsesPerNeuron_ * countNeurons_; i++) {
        if (W_[i] != 0.0) {
            cout << "lastSpike[" << i << "] = " << lastSpike_[i];
            cout << " r: " << r_[i];
            cout << " u: " << u_[i];
            cout << " D: " << D_[i];
            cout << " U: " << U_[i];
            cout << " F: " << F_[i] << endl;
        }
    }
}

