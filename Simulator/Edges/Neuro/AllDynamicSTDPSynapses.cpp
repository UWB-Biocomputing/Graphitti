/**
 * @file AllDynamicSTDPSynapses.cpp
 *
 * @ingroup Simulator/Edges
 * 
 * @brief A container of all dynamic STDP synapse data
 */

#include "AllDynamicSTDPSynapses.h"

AllDynamicSTDPSynapses::AllDynamicSTDPSynapses() : AllSTDPSynapses() {
    lastSpike_ = nullptr;
    r_ = nullptr;
    u_ = nullptr;
    D_ = nullptr;
    U_ = nullptr;
    F_ = nullptr;
}

AllDynamicSTDPSynapses::AllDynamicSTDPSynapses(const int numVertices, const int maxEdges) :
      AllSTDPSynapses(numVertices, maxEdges) {
    setupEdges(numVertices, maxEdges);
}

AllDynamicSTDPSynapses::~AllDynamicSTDPSynapses() {
    BGSIZE maxTotalSynapses = maxEdgesPerVertex_ * countVertices_;

    if (maxTotalSynapses != 0) {
        delete[] lastSpike_;
        delete[] r_;
        delete[] u_;
        delete[] D_;
        delete[] U_;
        delete[] F_;
    }

    lastSpike_ = nullptr;
    r_ = nullptr;
    u_ = nullptr;
    D_ = nullptr;
    U_ = nullptr;
    F_ = nullptr;
}

///  Setup the internal structure of the class (allocate memories and initialize them).
void AllDynamicSTDPSynapses::setupEdges() {
    setupEdges(Simulator::getInstance().getDeltaT(), Simulator::getInstance().getMaxEdgesPerVertex());
}

///  Setup the internal structure of the class (allocate memories and initialize them).
/// 
///  @param  numVertices   Total number of vertices in the network.
///  @param  maxEdges  Maximum number of synapses per neuron.
void AllDynamicSTDPSynapses::setupEdges(const int numVertices, const int maxEdges) {
    AllSTDPSynapses::setupEdges(numVertices, maxEdges);

    BGSIZE maxTotalSynapses = maxEdges * numVertices;

    if (maxTotalSynapses != 0) {
        lastSpike_ = new uint64_t[maxTotalSynapses];
        r_ = new BGFLOAT[maxTotalSynapses];
        u_ = new BGFLOAT[maxTotalSynapses];
        D_ = new BGFLOAT[maxTotalSynapses];
        U_ = new BGFLOAT[maxTotalSynapses];
        F_ = new BGFLOAT[maxTotalSynapses];
    }
}

///  Prints out all parameters to logging file.
///  Registered to OperationManager as Operation::printParameters
void AllDynamicSTDPSynapses::printParameters() const {
   AllSTDPSynapses::printParameters();

   LOG4CPLUS_DEBUG(fileLogger_, "\n\t---AllDynamicSTDPSynapses Parameters---" << endl
   << "\tEdges type: AllDynamicSTDPSynapses" << endl << endl);
}

///  Sets the data for Synapse to input's data.
///
///  @param  input  istream to read from.
///  @param  iEdg   Index of the synapse to set.
void AllDynamicSTDPSynapses::readEdge(istream &input, const BGSIZE iEdg) {
    AllSTDPSynapses::readEdge(input, iEdg);

    // input.ignore() so input skips over end-of-line characters.
    input >> lastSpike_[iEdg];
    input.ignore();
    input >> r_[iEdg];
    input.ignore();
    input >> u_[iEdg];
    input.ignore();
    input >> D_[iEdg];
    input.ignore();
    input >> U_[iEdg];
    input.ignore();
    input >> F_[iEdg];
    input.ignore();
}

///  Write the synapse data to the stream.
///
///  @param  output  stream to print out to.
///  @param  iEdg    Index of the synapse to print out.
void AllDynamicSTDPSynapses::writeEdge(ostream &output, const BGSIZE iEdg) const {
    AllSTDPSynapses::writeEdge(output, iEdg);

    output << lastSpike_[iEdg] << ends;
    output << r_[iEdg] << ends;
    output << u_[iEdg] << ends;
    output << D_[iEdg] << ends;
    output << U_[iEdg] << ends;
    output << F_[iEdg] << ends;
}

///  Reset time varying state vars and recompute decay.
///
///  @param  iEdg            Index of the synapse to set.
///  @param  deltaT          Inner simulation step duration
void AllDynamicSTDPSynapses::resetEdge(const BGSIZE iEdg, const BGFLOAT deltaT) {
    AllSTDPSynapses::resetEdge(iEdg, deltaT);

    u_[iEdg] = DEFAULT_U;
    r_[iEdg] = 1.0;
    lastSpike_[iEdg] = ULONG_MAX;
}

///  Create a Synapse and connect it to the model.
///
///  @param  iEdg        Index of the synapse to set.
///  @param  srcVertex   Coordinates of the source Neuron.
///  @param  destVertex  Coordinates of the destination Neuron.
///  @param  sumPoint    Summation point address.
///  @param  deltaT      Inner simulation step duration.
///  @param  type        Type of the Synapse to create.
void AllDynamicSTDPSynapses::createEdge(const BGSIZE iEdg, int srcVertex, int destVertex, BGFLOAT *sumPoint,
                                           const BGFLOAT deltaT, edgeType type) {
    AllSTDPSynapses::createEdge(iEdg, srcVertex, destVertex, sumPoint, deltaT, type);

    U_[iEdg] = DEFAULT_U;

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

    this->U_[iEdg] = U;
    this->D_[iEdg] = D;
    this->F_[iEdg] = F;
}

#if !defined(USE_GPU)

///  Calculate the post synapse response after a spike.
///
///  @param  iEdg        Index of the synapse to set.
///  @param  deltaT      Inner simulation step duration.
void AllDynamicSTDPSynapses::changePSR(const BGSIZE iEdg, const BGFLOAT deltaT) {
    BGFLOAT &psr = this->psr_[iEdg];
    BGFLOAT &W = this->W_[iEdg];
    BGFLOAT &decay = this->decay_[iEdg];
    uint64_t &lastSpike = this->lastSpike_[iEdg];
    BGFLOAT &r = this->r_[iEdg];
    BGFLOAT &u = this->u_[iEdg];
    BGFLOAT &D = this->D_[iEdg];
    BGFLOAT &F = this->F_[iEdg];
    BGFLOAT &U = this->U_[iEdg];

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

///  Prints SynapsesProps data.
void AllDynamicSTDPSynapses::printSynapsesProps() const {
    AllSTDPSynapses::printSynapsesProps();
    for (int i = 0; i < maxEdgesPerVertex_ * countVertices_; i++) {
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

