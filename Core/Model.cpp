#include "Model.h"
#include "tinyxml.h"
#include "ParseParamError.h"
#include "Util.h"
#include "ConnGrowth.h"

/// Constructor
/// ToDo: Stays the same right now, change further in refactor
Model::Model(Connections *conns, IAllNeurons *neurons, IAllSynapses *synapses, Layout *layout) :
    m_read_params(0),
    m_conns(conns),
    m_neurons(neurons),
    m_synapses(synapses),
    m_layout(layout),
    m_synapseIndexMap(NULL)
{
    simulator = Simulator::getInstance();
}

/// Destructor
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

/// Save simulation results to an output destination.
void Model::saveData()
{
    if (simulator->getSimRecorder() != NULL)
    {
        simulator->getSimRecorder()->saveSimData(*m_neurons);
    }
}

/// Creates all the Neurons and generates data for them.
void Model::createAllNeurons()
{
    DEBUG(cerr << "\nAllocating neurons..." << endl;)

    // init neuron's map with layout OLD
    //m_layout->generateNeuronTypeMap(sim_info->totalNeurons);
    //m_layout->initStarterMap(sim_info->totalNeurons);

    // init neuron's map with layout NEW
    // m_layout is a reference in model
    m_layout->generateNeuronTypeMap(simulator->getTotalNeurons());
    m_layout->initStarterMap(simulator->getTotalNeurons());

    // set their specific types
    m_neurons->createAllNeurons(m_layout);

    DEBUG(cerr << "Done initializing neurons..." << endl;)
}

/// Sets up the Simulation.
/// ToDo: find siminfo actual things being passed through
void Model::setupSim()
{
    DEBUG(cerr << "\tSetting up neurons....";)
    m_neurons->setupNeurons();
    DEBUG(cerr << "done.\n\tSetting up synapses....";)
    m_synapses->setupSynapses();
#ifdef PERFORMANCE_METRICS
    // Start timer for initialization
    sim_info->short_timer.start();
#endif
    DEBUG(cerr << "done.\n\tSetting up layout....";)
    m_layout->setupLayout();
    DEBUG(cerr << "done." << endl;)
#ifdef PERFORMANCE_METRICS
    // Time to initialization (layout)
    t_host_initialization_layout += sim_info->short_timer.lap() / 1000000.0;
#endif
    // Init radii and rates history matrices with default values
    if (simulator->getSimRecorder() != NULL) {
        simulator->getSimRecorder()->initDefaultValues();
    }

    // Creates all the Neurons and generates data for them.
    createAllNeurons();

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
    m_synapses->createSynapseImap(m_synapseIndexMap);
}

/// Clean up the simulation.
void Model::cleanupSim()
{
    m_neurons->cleanupNeurons();
    m_synapses->cleanupSynapses();
    m_conns->cleanupConnections();
}

/// Log this simulation step.
void Model::logSimStep() const
{
    ConnGrowth* pConnGrowth = dynamic_cast<ConnGrowth*>(m_conns);
    if (pConnGrowth == NULL)
        return;

    cout << "format:\ntype,radius,firing rate" << endl;

    for (int y = 0; y < simulator->getHeight; y++) {
        stringstream ss;
        ss << fixed;
        ss.precision(1);

        for (int x = 0; x < simulator->getWidth; x++) {
            switch (m_layout->neuron_type_map[x + y * simulator->getWidth]) {
            case EXC:
                if (m_layout->starter_map[x + y * simulator->getWidth])
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

            ss << " " << (*pConnGrowth->radii)[x + y * simulator->getWidth];

            if (x + 1 < simulator->getWidth) {
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

/// Update the simulation history of every epoch.
void Model::updateHistory()
{
    // Compile history information in every epoch
    if (simulator->getSimRecorder() != NULL) {
       simulator->getSimRecorder()->compileHistories(*m_neurons);
    }
}

/************************************************
 *  Accessors
 ***********************************************/

/// Get the IAllNeurons class object.
/// @return Pointer to the AllNeurons class object.  ToDo: make smart ptr
IAllNeurons* Model::getNeurons() {return m_neurons;}

/// Get the Connections class object.
/// @return Pointer to the Connections class object.  ToDo: make smart ptr
Connections* Model::getConnections() {return m_conns;}

/// Get the Layout class object.
/// @return Pointer to the Layout class object. ToDo: make smart ptr
Layout* Model::getLayout() {return m_layout;}

