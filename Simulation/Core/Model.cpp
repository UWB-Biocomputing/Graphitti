#include "Model.h"
#include "IRecorder.h"
#include "Connections.h"
#include "ConnGrowth.h"

/// Constructor
/// ToDo: Stays the same right now, change further in refactor
Model::Model(Connections *conns, Layout *layout) :
    read_params_(0),
    conns_(conns),
    layout_(layout),
    synapseIndexMap_(NULL)
{
}

/// Destructor todo: this will change
Model::~Model()
{
    if (conns_ != NULL) {
        delete conns_;
        conns_ = NULL;
    }

    if (neurons_ != NULL) {
        delete neurons_;
        neurons_ = NULL;
    }

    if (synapses_ != NULL) {
        delete synapses_;
        synapses_ = NULL;
    }

    if (layout_ != NULL) {
        delete layout_;
        layout_ = NULL;
    }

    if (synapseIndexMap_ != NULL) {
        delete synapseIndexMap_;
        synapseIndexMap_ = NULL;
    }
}

/// Save simulation results to an output destination.
// todo: recorder should be under model if not layout or connections
void Model::saveData()
{
    if (Simulator::getInstance().getSimRecorder() != NULL)
    {
       Simulator::getInstance().getSimRecorder()->saveSimData(*neurons_);
    }
}

/// Creates all the Neurons and generates data for them.
// todo: this is going to go away
void Model::createAllNeurons()
{
    DEBUG(cerr << "\nAllocating neurons..." << endl;)

    layout_->generateNeuronTypeMap(Simulator::getInstance().getTotalNeurons());
    layout_->initStarterMap(Simulator::getInstance().getTotalNeurons());

    // set their specific types
    // todo: neurons_
    neurons_->createAllNeurons(layout_);

    DEBUG(cerr << "Done initializing neurons..." << endl;)
}

/// Sets up the Simulation.
/// ToDo: find siminfo actual things being passed through
// todo: to be setup: tell layouts and connections to setup. will setup neurons/synapses.
// todo: setup recorders.
void Model::setupSim()
{
    DEBUG(cerr << "\tSetting up neurons....";)
    neurons_->setupNeurons();
    DEBUG(cerr << "done.\n\tSetting up synapses....";)
    synapses_->setupSynapses();
#ifdef PERFORMANCE_METRICS
    // Start timer for initialization
    Simulator::getInstance.short_timer.start();
#endif
    DEBUG(cerr << "done.\n\tSetting up layout....";)
    layout_->setupLayout();
    DEBUG(cerr << "done." << endl;)
#ifdef PERFORMANCE_METRICS
    // Time to initialization (layout)
    t_host_initialization_layout += Simulator::getInstance().short_timer.lap() / 1000000.0;
#endif
    // Init radii and rates history matrices with default values
    if (Simulator::getInstance().getSimRecorder() != NULL) {
        Simulator::getInstance().getSimRecorder()->initDefaultValues();
    }

    // Creates all the Neurons and generates data for them.
    createAllNeurons();

#ifdef PERFORMANCE_METRICS
    // Start timer for initialization
    Simulator::getInstance().short_timer.start();
#endif
    conns_->setupConnections(layout_, neurons_, synapses_);
#ifdef PERFORMANCE_METRICS
    // Time to initialization (connections)
    t_host_initialization_connections += Simulator::getInstance().short_timer.lap() / 1000000.0;
#endif

    // create a synapse index map
    synapses_->createSynapseImap(synapseIndexMap_);
}

/// Clean up the simulation.
void Model::cleanupSim()
{
    neurons_->cleanupNeurons();
    synapses_->cleanupSynapses();
    conns_->cleanupConnections();
}

/// Log this simulation step.
void Model::logSimStep() const
{
    ConnGrowth* pConnGrowth = dynamic_cast<ConnGrowth*>(conns_);
    if (pConnGrowth == NULL)
        return;

    cout << "format:\ntype,radius,firing rate" << endl;

    for (int y = 0; y < Simulator::getInstance().getHeight(); y++) {
        stringstream ss;
        ss << fixed;
        ss.precision(1);

        for (int x = 0; x < Simulator::getInstance().getWidth(); x++) {
            switch (layout_->neuron_type_map[x + y * Simulator::getInstance().getWidth()]) {
            case EXC:
                if (layout_->starter_map[x + y * Simulator::getInstance().getWidth()])
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

            ss << " " << (*pConnGrowth->radii)[x + y * Simulator::getInstance().getWidth()];

            if (x + 1 < Simulator::getInstance().getWidth()) {
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
    if (Simulator::getInstance().getSimRecorder() != NULL) {
       Simulator::getInstance().getSimRecorder()->compileHistories(*neurons_);
    }
}

/************************************************
 *  Accessors
 ***********************************************/

/// Get the Connections class object.
/// @return Pointer to the Connections class object.  ToDo: make smart ptr
Connections* Model::getConnections() {return conns_;}

/// Get the Layout class object.
/// @return Pointer to the Layout class object. ToDo: make smart ptr
Layout* Model::getLayout() {return layout_;}
