#include "FixedLayout.h"
#include "ParseParamError.h"
#include "Util.h"

FixedLayout::FixedLayout() : Layout() {
}

FixedLayout::~FixedLayout() {
}

/*
 *  Prints out all parameters of the layout to console.
 */
void FixedLayout::printParameters() const {
   Layout::printParameters();

   cout << "Layout parameters:" << endl;

   cout << "\tEndogenously active neuron positions: ";
   for (BGSIZE i = 0; i < numEndogenouslyActiveNeurons_; i++) {
      cout << endogenouslyActiveNeuronList_[i] << " ";
   }

   cout << endl;

   cout << "\tInhibitory neuron positions: ";
   for (BGSIZE i = 0; i < inhibitoryNeuronLayout_.size(); i++) {
      cout << inhibitoryNeuronLayout_[i] << " ";
   }

   cout << endl;

   cout << "\tProbed neuron positions: ";
   for (BGSIZE i = 0; i < probedNeuronList_.size(); i++) {
      cout << probedNeuronList_[i] << " ";
   }

   cout << endl;
}

/*
 *  Creates a randomly ordered distribution with the specified numbers of neuron types.
 *  @param  num_neurons number of the neurons to have in the type map.
 *  @return a flat vector (to map to 2-d [x,y] = [i % m_width, i / m_width])
 */
void FixedLayout::generateNeuronTypeMap(int num_neurons) {
   Layout::generateNeuronTypeMap(num_neurons);

   int num_inhibitory_neurons = inhibitoryNeuronLayout_.size();
   int num_excititory_neurons = num_neurons - num_inhibitory_neurons;

   DEBUG(cout << "Total neurons: " << num_neurons << endl;)
   DEBUG(cout << "Inhibitory Neurons: " << num_inhibitory_neurons << endl;)
   DEBUG(cout << "Excitatory Neurons: " << num_excititory_neurons << endl;)

   for (int i = 0; i < num_inhibitory_neurons; i++) {
      assert(inhibitoryNeuronLayout_.at(i) < num_neurons);
      neuronTypeMap_[inhibitoryNeuronLayout_.at(i)] = INH;
   }

   DEBUG(cout << "Done initializing neuron type map" << endl;);
}

/*
 *  Populates the starter map.
 *  Selects \e numStarter excitory neurons and converts them into starter neurons.
 *  @param  num_neurons number of neurons to have in the map.
 */
void FixedLayout::initStarterMap(const int num_neurons) {
   Layout::initStarterMap(num_neurons);

   for (BGSIZE i = 0; i < numEndogenouslyActiveNeurons_; i++) {
      assert(endogenouslyActiveNeuronList_.at(i) < num_neurons);
      starterMap_[endogenouslyActiveNeuronList_.at(i)] = true;
   }
}
