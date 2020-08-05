#include "CPUSpikingModel.h"
#include "Simulator.h"
#include "AllDSSynapses.h"

/// Constructor
CPUSpikingModel::CPUSpikingModel(
      Connections *conns,
      Layout *layout) :
      Model(conns, layout) {
}

/// Destructor
CPUSpikingModel::~CPUSpikingModel() {
   //Let Model base class handle de-allocation
}

/// Sets up the Simulation.
void CPUSpikingModel::setupSim() {
   Model::setupSim();
   // Create a normalized random number generator
   rgNormrnd.push_back(new Norm(0, 1, Simulator::getInstance().getSeed()));
}

/// Advance everything in the model one time step.
void CPUSpikingModel::advance() {
   // ToDo: look at pointer v no pointer in params - to change
   // dereferencing the ptr, lose late binding -- look into changing!
   layout_->getNeurons()->advanceNeurons(*conns_->getSynapses(), synapseIndexMap_);
   conns_->getSynapses()->advanceSynapses(layout_->getNeurons(), synapseIndexMap_);
}

/// Update the connection of all the Neurons and Synapses of the simulation.
void CPUSpikingModel::updateConnections() {
   // Update Connections data
   if (conns_->updateConnections(*layout_->getNeurons(), layout_.get())) {
      conns_->updateSynapsesWeights(
            Simulator::getInstance().getTotalNeurons(),
            *layout_->getNeurons(),
            *conns_->getSynapses(),
            layout_.get());
      // create synapse inverse map
      conns_->getSynapses()->createSynapseImap(synapseIndexMap_);
   }
}

/// Copy GPU Synapse data to CPU. (Inheritance, no implem)
void CPUSpikingModel::copyGPUtoCPU() {
   cerr << "ERROR: CPUSpikingModel::copyGPUtoCPU() was called." << endl;
   exit(EXIT_FAILURE);
}

/// Copy CPU Synapse data to GPU. (Inheritance, no implem, GPUModel has implem)
void CPUSpikingModel::copyCPUtoGPU() {
   cerr << "ERROR: CPUSpikingModel::copyCPUtoGPU() was called." << endl;
   exit(EXIT_FAILURE);
}