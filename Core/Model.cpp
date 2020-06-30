#include "Model.h"

#include "tinyxml.h"

#include "ParseParamError.h"
#include "Util.h"
#include "ConnGrowth.h"


/*
 *  Constructor
 */
Model::Model(Connections *conns, IAllNeurons *neurons, IAllSynapses *synapses, Layout *layout) :
    m_read_params(0),
    m_conns(conns),
    m_neurons(neurons),
    m_synapses(synapses),
    m_layout(layout),
    m_synapseIndexMap(NULL)
{
}

/*
 *  Destructor
 */
Model::~Model()
{
    if (m_conns != NULL) {
        delete m_conns;
        m_conns = NULL;
    }

    if (m_neurons != NULL) {
        delete m_neurons;
        m_neurons = NULL;
    }

    if (m_synapses != NULL) {
        delete m_synapses;
        m_synapses = NULL;
    }

    if (m_layout != NULL) {
        delete m_layout;
        m_layout = NULL;
    }

    if (m_synapseIndexMap != NULL) {
        delete m_synapseIndexMap;
        m_synapseIndexMap = NULL;
    }
}

/*
 *  Save simulation results to an output destination.
 *
 *  @param  sim_info    parameters for the simulation. 
 */
void Model::saveData(SimulationInfo *sim_info)
{
    if (sim_info->simRecorder != NULL) {
        sim_info->simRecorder->saveSimData(*m_neurons);
    }
}

/*
 *  Creates all the Neurons and generates data for them.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void Model::createAllNeurons(SimulationInfo *sim_info)
{
    DEBUG(cerr << "\nAllocating neurons..." << endl;)

    // init neuron's map with layout
    m_layout->generateNeuronTypeMap(sim_info->totalNeurons);
    m_layout->initStarterMap(sim_info->totalNeurons);

    // set their specific types
    m_neurons->createAllNeurons(sim_info, m_layout);

    DEBUG(cerr << "Done initializing neurons..." << endl;)
}

/*
 *  Sets up the Simulation.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void Model::setupSim(SimulationInfo *sim_info)
{
    DEBUG(cerr << "\tSetting up neurons....";)
    m_neurons->setupNeurons(sim_info);
    DEBUG(cerr << "done.\n\tSetting up synapses....";)
    m_synapses->setupSynapses(sim_info);
#ifdef PERFORMANCE_METRICS
    // Start timer for initialization
    sim_info->short_timer.start();
#endif
    DEBUG(cerr << "done.\n\tSetting up layout....";)
    m_layout->setupLayout(sim_info);
    DEBUG(cerr << "done." << endl;)
#ifdef PERFORMANCE_METRICS
    // Time to initialization (layout)
    t_host_initialization_layout += sim_info->short_timer.lap() / 1000000.0;
#endif

    // Init radii and rates history matrices with default values
    if (sim_info->simRecorder != NULL) {
        sim_info->simRecorder->initDefaultValues();
    }

    // Creates all the Neurons and generates data for them.
    createAllNeurons(sim_info);

#ifdef PERFORMANCE_METRICS
    // Start timer for initialization
    sim_info->short_timer.start();
#endif
    m_conns->setupConnections(sim_info, m_layout, m_neurons, m_synapses);
#ifdef PERFORMANCE_METRICS
    // Time to initialization (connections)
    t_host_initialization_connections += sim_info->short_timer.lap() / 1000000.0;
#endif

    // create a synapse index map 
    m_synapses->createSynapseImap(m_synapseIndexMap, sim_info);
}

/*
 *  Clean up the simulation.
 *
 *  @param  sim_info    SimulationInfo to refer.
 */
void Model::cleanupSim(SimulationInfo *sim_info)
{
    m_neurons->cleanupNeurons();
    m_synapses->cleanupSynapses();
    m_conns->cleanupConnections();
}

/*
 *  Log this simulation step.
 *
 *  @param  sim_info    SimulationInfo to reference.
 */
void Model::logSimStep(const SimulationInfo *sim_info) const
{
    ConnGrowth* pConnGrowth = dynamic_cast<ConnGrowth*>(m_conns);
    if (pConnGrowth == NULL)
        return;

    cout << "format:\ntype,radius,firing rate" << endl;

    for (int y = 0; y < sim_info->height; y++) {
        stringstream ss;
        ss << fixed;
        ss.precision(1);

        for (int x = 0; x < sim_info->width; x++) {
            switch (m_layout->neuron_type_map[x + y * sim_info->width]) {
            case EXC:
                if (m_layout->starter_map[x + y * sim_info->width])
                    ss << "s";
                else
                    ss << "e";
                break;
            case INH:
                ss << "i";
                break;
            case NTYPE_UNDEF:
                assert(false);
                break;
            }

            ss << " " << (*pConnGrowth->radii)[x + y * sim_info->width];

            if (x + 1 < sim_info->width) {
                ss.width(2);
                ss << "|";
                ss.width(2);
            }
        }

        ss << endl;

        for (int i = ss.str().length() - 1; i >= 0; i--) {
            ss << "_";
        }

        ss << endl;
        cout << ss.str();
    }
}

/*
 *  Update the simulation history of every epoch.
 *
 *  @param  sim_info    SimulationInfo to refer from.
 */
void Model::updateHistory(const SimulationInfo *sim_info)
{
    // Compile history information in every epoch
    if (sim_info->simRecorder != NULL) {
        sim_info->simRecorder->compileHistories(*m_neurons);
    }
}

/*
 *  Get the IAllNeurons class object.
 *
 *  @return Pointer to the AllNeurons class object.
 */
IAllNeurons* Model::getNeurons()
{
    return m_neurons;
}

/*
 *  Get the Connections class object.
 *
 *  @return Pointer to the Connections class object.
 */
Connections* Model::getConnections()
{
    return m_conns;
}

/*
 *  Get the Layout class object.
 *
 *  @return Pointer to the Layout class object.
 */
Layout* Model::getLayout()
{
    return m_layout;
}

