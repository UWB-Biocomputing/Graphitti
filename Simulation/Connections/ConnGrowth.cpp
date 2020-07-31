#include "ConnGrowth.h"
#include "ParseParamError.h"
#include "IAllSynapses.h"
#include "XmlGrowthRecorder.h"
#include "AllSpikingNeurons.h"
#include "Matrix/CompleteMatrix.h"
#include "Matrix/Matrix.h"
#include "Matrix/VectorMatrix.h"

#ifdef USE_HDF5
#include "Hdf5GrowthRecorder.h"
#endif

/* ------------- CONNECTIONS STRUCT ------------ *\
 * Below all of the resources for the various
 * connections are instantiated and initialized.
 * All of the allocation for memory is done in the
 * constructor’s parameters and not in the body of
 * the function. Once all memory has been allocated
 * the constructor fills in known information
 * into “radii” and “rates”.
\* --------------------------------------------- */
// TODO comment
/* ------------------- ERROR ------------------- *\
 * terminate called after throwing an instance of 'std::bad_alloc'
 *      what():  St9bad_alloc
 * ------------------- CAUSE ------------------- *|
 * As simulations expand in size the number of
 * neurons in total increases exponentially. When
 * using a MATRIX_TYPE = “complete” the amount of
 * used memory increases by another order of magnitude.
 * Once enough memory is used no more memory can be
 * allocated and a “bsd_alloc” will be thrown.
 * The following members of the connection constructor
 * consume equally vast amounts of memory as the
 * simulation sizes grow:
 *      - W             - radii
 *      - rates         - dist2
 *      - delta         - dist
 *      - areai
 * ----------------- 1/25/14 ------------------- *|
 * Currently when running a simulation of sizes
 * equal to or greater than 100 * 100 the above
 * error is thrown. After some testing we have
 * determined that this is a hardware dependent
 * issue, not software. We are also looking into
 * switching matrix types from "complete" to
 * "sparce". If successful it is possible the
 * problematic matricies mentioned above will use
 * only 1/250 of their current space.
\* --------------------------------------------- */
ConnGrowth::ConnGrowth() : Connections()
{
    W = NULL;
    radii = NULL;
    rates = NULL;
    delta = NULL;
    area = NULL;
    outgrowth = NULL;
    deltaR = NULL;
    radiiSize = 0;
}

ConnGrowth::~ConnGrowth()
{
    cleanupConnections();
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  layout    Layout information of the neunal network.
 *  @param  neurons   The Neuron list to search from.
 *  @param  synapses  The Synapse list to search from.
 */
void ConnGrowth::setupConnections(Layout *layout, IAllNeurons *neurons, IAllSynapses *synapses)
{
    int num_neurons = Simulator::getInstance().getTotalNeurons();
    radiiSize = num_neurons;

    W = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons, 0);
    radii = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons, m_growth.startRadius);
    rates = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons, 0);
    delta = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons);
    area = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons, 0);
    outgrowth = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons);
    deltaR = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons);

    // Init connection frontier distance change matrix with the current distances
    (*delta) = (*layout->dist);
}

/*
 *  Cleanup the class (deallocate memories).
 */
void ConnGrowth::cleanupConnections()
{
    if (W != NULL) delete W;
    if (radii != NULL) delete radii;
    if (rates != NULL) delete rates;
    if (delta != NULL) delete delta;
    if (area != NULL) delete area;
    if (outgrowth != NULL) delete outgrowth;
    if (deltaR != NULL) delete deltaR;

    W = NULL;
    radii = NULL;
    rates = NULL;
    delta = NULL;
    area = NULL;
    outgrowth = NULL;
    deltaR = NULL;
    radiiSize = 0;
}

/*
 *  Checks the number of required parameters.
 *
 * @return true if all required parameters were successfully read, false otherwise.
 */
bool ConnGrowth::checkNumParameters()
{
    return (nParams >= 1);
}

/*
 *  Prints out all parameters of the connections to ostream.
 *
 *  @param  output  ostream to send output to.
 */
void ConnGrowth::printParameters(ostream &output) const
{
    output << "Growth parameters: " << endl
           << "\tepsilon: " << m_growth.epsilon
           << ", beta: " << m_growth.beta
           << ", rho: " << m_growth.rho
           << ", targetRate: " << m_growth.targetRate << "," << endl
           << "\tminRadius: " << m_growth.minRadius
           << ", startRadius: " << m_growth.startRadius
           << endl;

}

/*
 *  Update the connections status in every epoch.
 *
 *  @param  neurons  The Neuron list to search from.
 *  @param  layout   Layout information of the neunal network.
 *  @return true if successful, false otherwise.
 */
bool ConnGrowth::updateConnections(IAllNeurons &neurons, Layout *layout)
{
    // Update Connections data
    updateConns(neurons);
 
    // Update the distance between frontiers of Neurons
    updateFrontiers(Simulator::getInstance().getTotalNeurons(), layout);

    // Update the areas of overlap in between Neurons
    updateOverlap(Simulator::getInstance().getTotalNeurons(), layout);

    return true;
}

/*
 *  Calculates firing rates, neuron radii change and assign new values.
 *
 *  @param  neurons  The Neuron list to search from.
 */
void ConnGrowth::updateConns(IAllNeurons &neurons)
{
    AllSpikingNeurons &spNeurons = dynamic_cast<AllSpikingNeurons&>(neurons);

    // Calculate growth cycle firing rate for previous period
    int max_spikes = static_cast<int> (Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getMaxFiringRate());
    for (int i = 0; i < Simulator::getInstance().getTotalNeurons(); i++) {
        // Calculate firing rate
        assert(spNeurons.spikeCount[i] < max_spikes);
        (*rates)[i] = spNeurons.spikeCount[i] / Simulator::getInstance().getEpochDuration();
    }

    // compute neuron radii change and assign new values
    (*outgrowth) = 1.0 - 2.0 / (1.0 + exp((m_growth.epsilon - *rates / m_growth.maxRate) / m_growth.beta));
    (*deltaR) = Simulator::getInstance().getEpochDuration() * m_growth.rho * *outgrowth;
    (*radii) += (*deltaR);
}

/*
 *  Update the distance between frontiers of Neurons.
 *
 *  @param  num_neurons Number of neurons to update.
 *  @param  layout      Layout information of the neunal network.
 */
void ConnGrowth::updateFrontiers(const int num_neurons, Layout *layout)
{
    DEBUG(cout << "Updating distance between frontiers..." << endl;)
    // Update distance between frontiers
    for (int unit = 0; unit < num_neurons - 1; unit++) {
        for (int i = unit + 1; i < num_neurons; i++) {
            (*delta)(unit, i) = (*layout->dist)(unit, i) - ((*radii)[unit] + (*radii)[i]);
            (*delta)(i, unit) = (*delta)(unit, i);
        }
    }
}

/*
 *  Update the areas of overlap in between Neurons.
 *
 *  @param  num_neurons Number of Neurons to update.
 *  @param  layout      Layout information of the neunal network.
 */
void ConnGrowth::updateOverlap(BGFLOAT num_neurons, Layout *layout)
{
    DEBUG(cout << "computing areas of overlap" << endl;)

    // Compute areas of overlap; this is only done for overlapping units
    for (int i = 0; i < num_neurons; i++) {
        for (int j = 0; j < num_neurons; j++) {
                (*area)(i, j) = 0.0;

                if ((*delta)(i, j) < 0) {
                        BGFLOAT lenAB = (*layout->dist)(i, j);
                        BGFLOAT r1 = (*radii)[i];
                        BGFLOAT r2 = (*radii)[j];

                    if (lenAB + min(r1, r2) <= max(r1, r2)) {
                        (*area)(i, j) = pi * min(r1, r2) * min(r1, r2); // Completely overlapping unit
                        
#ifdef LOGFILE
                        logFile << "Completely overlapping (i, j, r1, r2, area): "
                            << i << ", " << j << ", " << r1 << ", " << r2 << ", " << *pAarea(i, j) << endl;
#endif // LOGFILE
                        } else {
                                // Partially overlapping unit
                                BGFLOAT lenAB2 = (*layout->dist2)(i, j);
                                BGFLOAT r12 = r1 * r1;
                                BGFLOAT r22 = r2 * r2;

                                BGFLOAT cosCBA = (r22 + lenAB2 - r12) / (2.0 * r2 * lenAB);
                                BGFLOAT cosCAB = (r12 + lenAB2 - r22) / (2.0 * r1 * lenAB);
                            
                                if(fabs(cosCBA) >= 1.0 || fabs(cosCAB) >= 1.0) {
                                    (*area)(i,j) = 0.0;
                                } else {

                                    BGFLOAT angCBA = acos(cosCBA);
                                    BGFLOAT angCBD = 2.0 * angCBA;

                                    BGFLOAT angCAB = acos(cosCAB);
                                    BGFLOAT angCAD = 2.0 * angCAB;

                                    (*area)(i, j) = 0.5 * (r22 * (angCBD - sin(angCBD)) + r12 * (angCAD - sin(angCAD)));
                                }
                        }
                }
        }
    }
}

#if !defined(USE_GPU)
/*
 *  Update the weight of the Synapses in the simulation.
 *  Note: Platform Dependent.
 *
 *  @param  num_neurons Number of neurons to update.
 *  @param  ineurons    The Neuron list to search from.
 *  @param  isynapses   The Synapse list to search from.
 */
void ConnGrowth::updateSynapsesWeights(const int num_neurons, IAllNeurons &ineurons, IAllSynapses &isynapses, Layout *layout)
{
    AllNeurons &neurons = dynamic_cast<AllNeurons&>(ineurons);
    AllSynapses &synapses = dynamic_cast<AllSynapses&>(isynapses);

    // For now, we just set the weights to equal the areas. We will later
    // scale it and set its sign (when we index and get its sign).
    (*W) = (*area);

    int adjusted = 0;
    int could_have_been_removed = 0; // TODO: use this value
    int removed = 0;
    int added = 0;

    DEBUG(cout << "adjusting weights" << endl;)

    // Scale and add sign to the areas
    // visit each neuron 'a'
    for (int src_neuron = 0; src_neuron < num_neurons; src_neuron++) {
        // and each destination neuron 'b'
        for (int dest_neuron = 0; dest_neuron < num_neurons; dest_neuron++) {
            // visit each synapse at (xa,ya)
            bool connected = false;
            synapseType type = layout->synType(src_neuron, dest_neuron);

            // for each existing synapse
            BGSIZE synapse_counts = synapses.synapse_counts[dest_neuron];
            BGSIZE synapse_adjusted = 0;
            BGSIZE iSyn = Simulator::getInstance().getMaxSynapsesPerNeuron() * dest_neuron;
            for (BGSIZE synapse_index = 0; synapse_adjusted < synapse_counts; synapse_index++, iSyn++) {
                if (synapses.in_use[iSyn] == true) {
                    // if there is a synapse between a and b
                    if (synapses.sourceNeuronIndex[iSyn] == src_neuron) {
                        connected = true;
                        adjusted++;
                        // adjust the strength of the synapse or remove
                        // it from the synapse map if it has gone below
                        // zero.
                        if ((*W)(src_neuron, dest_neuron) <= 0) {
                            removed++;
                            synapses.eraseSynapse(dest_neuron, iSyn);
                        } else {
                            // adjust
                            // SYNAPSE_STRENGTH_ADJUSTMENT is 1.0e-8;
                            synapses.W[iSyn] = (*W)(src_neuron, dest_neuron) *
                                synapses.synSign(type) * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;

                            DEBUG_MID(cout << "weight of rgSynapseMap" <<
                                   "[" <<synapse_index<<"]: " <<
                                   synapses.W[iSyn] << endl;);
                        }
                    }
                    synapse_adjusted++;
                }
            }

            // if not connected and weight(a,b) > 0, add a new synapse from a to b
            if (!connected && ((*W)(src_neuron, dest_neuron) > 0)) {

                // locate summation point
                BGFLOAT* sum_point = &( neurons.summation_map[dest_neuron] );
                added++;

                BGSIZE iSyn;
                synapses.addSynapse(iSyn, type, src_neuron, dest_neuron, sum_point, Simulator::getInstance().getDeltaT());
                synapses.W[iSyn] = (*W)(src_neuron, dest_neuron) * synapses.synSign(type) * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;

            }
        }
    }

    DEBUG (cout << "adjusted: " << adjusted << endl;)
    DEBUG (cout << "could have been removed (TODO: calculate this): " << could_have_been_removed << endl;)
    DEBUG (cout << "removed: " << removed << endl;)
    DEBUG (cout << "added: " << added << endl << endl << endl;)
}
#endif // !USE_GPU

/*
 *  Creates a recorder class object for the connection.
 *  This function tries to create either Xml recorder or
 *  Hdf5 recorder based on the extension of the file name.
 *
 *  @return Pointer to the recorder class object.
 */
IRecorder* ConnGrowth::createRecorder()
{
    // create & init simulation recorder
    IRecorder* simRecorder = NULL;
    if (Simulator::getInstance().getStateOutputFileName().find(".xml") != string::npos) {
       simRecorder = new XmlRecorder();
    }
#ifdef USE_HDF5
    else if (simInfo->stateOutputFileName.find(".h5") != string::npos) {
        simRecorder = new Hdf5GrowthRecorder(simInfo);
    }
#endif // USE_HDF5
    else {
        return NULL;
    }
    if (simRecorder != NULL) {
        simRecorder->init(Simulator::getInstance().getStateOutputFileName());
    }

    return simRecorder;
}

/**
 *  Prints radii 
 */
void ConnGrowth::printRadii() const {
	for (int i = 0; i < radiiSize; i++) {
		cout << "radii[" << i << "] = " << (*radii)[i] << endl;
	}
}