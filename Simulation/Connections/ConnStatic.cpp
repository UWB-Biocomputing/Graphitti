#include "ConnStatic.h"
#include "ParseParamError.h"
#include "IAllSynapses.h"
#include "XmlRecorder.h"
#ifdef USE_HDF5
#include "Hdf5Recorder.h"
#endif
#include <algorithm>

ConnStatic::ConnStatic() : Connections()
{
    m_threshConnsRadius = 0;
    m_nConnsPerNeuron = 0;
    m_pRewiring = 0;
}

ConnStatic::~ConnStatic()
{
    cleanupConnections();
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *  Initialize the small world network characterized by parameters: 
 *  number of maximum connections per neurons, connection radius threshold, and
 *  small-world rewiring probability.
 *
 *  @param  layout    Layout information of the neunal network.
 *  @param  neurons   The Neuron list to search from.
 *  @param  synapses  The Synapse list to search from.
 */
void ConnStatic::setupConnections(Layout *layout, IAllNeurons *neurons, IAllSynapses *synapses)
{
    int num_neurons = Simulator::getInstance().getTotalNeurons();
    vector<DistDestNeuron> distDestNeurons[num_neurons];

    int added = 0;

    DEBUG(cout << "Initializing connections" << endl;)

    for (int src_neuron = 0; src_neuron < num_neurons; src_neuron++) {
        distDestNeurons[src_neuron].clear();

        // pick the connections shorter than threshConnsRadius
        for (int dest_neuron = 0; dest_neuron < num_neurons; dest_neuron++) {
            if (src_neuron != dest_neuron) {
                BGFLOAT dist = (*layout->dist)(src_neuron, dest_neuron);
                if (dist <= m_threshConnsRadius) {
                    DistDestNeuron distDestNeuron;
                    distDestNeuron.dist = dist;
                    distDestNeuron.dest_neuron = dest_neuron;
                    distDestNeurons[src_neuron].push_back(distDestNeuron);
                }
            }
        }

        // sort ascendant
        sort(distDestNeurons[src_neuron].begin(), distDestNeurons[src_neuron].end());
        // pick the shortest m_nConnsPerNeuron connections
        for (BGSIZE i = 0; i < distDestNeurons[src_neuron].size() && (int)i < m_nConnsPerNeuron; i++) {
            int dest_neuron = distDestNeurons[src_neuron][i].dest_neuron;
            synapseType type = layout->synType(src_neuron, dest_neuron);
            BGFLOAT* sum_point = &( dynamic_cast<AllNeurons*>(neurons)->summation_map[dest_neuron] );

            DEBUG_MID (cout << "source: " << src_neuron << " dest: " << dest_neuron << " dist: " << distDestNeurons[src_neuron][i].dist << endl;)

            BGSIZE iSyn;
            synapses->addSynapse(iSyn, type, src_neuron, dest_neuron, sum_point, Simulator::getInstance().getDeltaT());
            added++;

            // set synapse weight
            // TODO: we need another synaptic weight distibution mode (normal distribution)
            if (synapses->synSign(type) > 0) {
                dynamic_cast<AllSynapses*>(synapses)->W[iSyn] = rng.inRange(m_excWeight[0], m_excWeight[1]);
            }
            else {
                dynamic_cast<AllSynapses*>(synapses)->W[iSyn] = rng.inRange(m_inhWeight[0], m_inhWeight[1]);
            } 
        }
    }

    int nRewiring = added * m_pRewiring;

    DEBUG(cout << "Rewiring connections: " << nRewiring << endl;)

    DEBUG (cout << "added connections: " << added << endl << endl << endl;)
}

/*
 *  Cleanup the class.
 */
void ConnStatic::cleanupConnections()
{
}

/*
 *  Checks the number of required parameters.
 *
 * @return true if all required parameters were successfully read, false otherwise.
 */
bool ConnStatic::checkNumParameters()
{
    return (nParams >= 2);
}

/*
 *  Prints out all parameters of the connections to ostream.
 *
 *  @param  output  ostream to send output to.
 */
void ConnStatic::printParameters(ostream &output) const
{
}

/*
 *  Creates a recorder class object for the connection.
 *  This function tries to create either Xml recorder or
 *  Hdf5 recorder based on the extension of the file name.
 *
 *  @return Pointer to the recorder class object.
 */
IRecorder* ConnStatic::createRecorder() {
    // create & init simulation recorder
    IRecorder* simRecorder = NULL;
    if (Simulator::getInstance().getStateOutputFileName().find(".xml") != string::npos) {
        simRecorder = new XmlRecorder();
    }
#ifdef USE_HDF5
    else if (Simulator::getInstance().getStateOutputFileName().find(".h5") != string::npos) {
        simRecorder = new Hdf5Recorder();
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
