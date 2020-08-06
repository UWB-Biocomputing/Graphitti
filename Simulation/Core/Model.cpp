#include "Model.h"
#include "IRecorder.h"
#include "Connections.h"
#include "ConnGrowth.h"

/// Constructor
/// ToDo: Stays the same right now, change further in refactor
Model::Model(Connections *conns, Layout *layout) :
      conns_(conns),
      layout_(layout)
      {}

/// Destructor todo: this will change
Model::~Model() {

}

/// Save simulation results to an output destination.
// todo: recorder should be under model if not layout or connections
void Model::saveData() {
   if (recorder_ != NULL) {
      recorder_->saveSimData(*layout_->getNeurons());
   }
}

/// Creates all the Neurons and generates data for them.
// todo: this is going to go away
void Model::createAllNeurons() {
   DEBUG(cerr << "\nAllocating neurons..." << endl;)

   layout_->generateNeuronTypeMap(Simulator::getInstance().getTotalNeurons());
   layout_->initStarterMap(Simulator::getInstance().getTotalNeurons());

   // set their specific types
   // todo: neurons_
   layout_->getNeurons()->createAllNeurons(layout_.get());

   DEBUG(cerr << "Done initializing neurons..." << endl;)
}

/// Sets up the Simulation.
/// ToDo: find siminfo actual things being passed through
// todo: to be setup: tell layouts and connections to setup. will setup neurons/synapses.
// todo: setup recorders.
void Model::setupSim() {
   DEBUG(cerr << "\tSetting up neurons....";)
   layout_->getNeurons()->setupNeurons();
   DEBUG(cerr << "done.\n\tSetting up synapses....";)
   conns_->getSynapses()->setupSynapses();
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
   if (recorder_ != NULL) {
      recorder_->initDefaultValues();
   }

   // Creates all the Neurons and generates data for them.
   createAllNeurons();

#ifdef PERFORMANCE_METRICS
   // Start timer for initialization
   Simulator::getInstance().short_timer.start();
#endif
   conns_->setupConnections(layout_.get(), layout_->getNeurons().get(), conns_->getSynapses().get());
#ifdef PERFORMANCE_METRICS
   // Time to initialization (connections)
   t_host_initialization_connections += Simulator::getInstance().short_timer.lap() / 1000000.0;
#endif

   // create a synapse index map
   conns_->getSynapses()->createSynapseImap(conns_->getSynapseIndexMap().get());
}

/// Clean up the simulation.
void Model::cleanupSim() {
   layout_->getNeurons()->cleanupNeurons();
   conns_->getSynapses()->cleanupSynapses();
   conns_->cleanupConnections();
}

/// Log this simulation step.
void Model::logSimStep() const {
   ConnGrowth *pConnGrowth = dynamic_cast<ConnGrowth *>(conns_.get());
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
void Model::updateHistory() {
   // Compile history information in every epoch
   if (recorder_ != nullptr) {
      recorder_->compileHistories(*layout_->getNeurons());
   }
}

/************************************************
 *  Accessors
 ***********************************************/

/// Get the Connections class object.
/// @return Pointer to the Connections class object.  ToDo: make smart ptr
shared_ptr<Connections> Model::getConnections() const { return conns_; }

/// Get the Layout class object.
/// @return Pointer to the Layout class object. ToDo: make smart ptr
shared_ptr<Layout> Model::getLayout() const { return layout_; }

shared_ptr<IRecorder> Model::getRecorder() const { return recorder_; }
