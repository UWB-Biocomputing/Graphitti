#include "DynamicLayout.h"
#include "ParseParamError.h"
#include "Util.h"

DynamicLayout::DynamicLayout() : Layout() {
}

DynamicLayout::~DynamicLayout() {
}

/*
 *  Prints out all parameters of the layout to console.
 */
void DynamicLayout::printParameters() const {
   Layout::printParameters();
   cout << "\tLayout type: Dynamic Layout" << endl;

   cout << "\tfrac_EXC:" << m_frac_excitatory_neurons << endl;
   cout << "\tStarter neurons:" << m_frac_starter_neurons << endl << endl;
}

/*
 *  Creates a randomly ordered distribution with the specified numbers of neuron types.
 *
 *  @param  num_neurons number of the neurons to have in the type map.
 */
void DynamicLayout::generateNeuronTypeMap(int num_neurons) {
   Layout::generateNeuronTypeMap(num_neurons);

   int num_excititory_neurons = (int) (m_frac_excitatory_neurons * num_neurons + 0.5);
   int num_inhibitory_neurons = num_neurons - num_excititory_neurons;

   DEBUG(cout << "Total neurons: " << num_neurons << endl;)
   DEBUG(cout << "Inhibitory Neurons: " << num_inhibitory_neurons << endl;)
   DEBUG(cout << "Excitatory Neurons: " << num_excititory_neurons << endl;)

   DEBUG(cout << endl << "Randomly selecting inhibitory neurons..." << endl;)

   int *rg_inhibitory_layout = new int[num_inhibitory_neurons];

   for (int i = 0; i < num_inhibitory_neurons; i++) {
      rg_inhibitory_layout[i] = i;
   }

   for (int i = num_inhibitory_neurons; i < num_neurons; i++) {
      int j = static_cast<int>(rng() * num_neurons);
      if (j < num_inhibitory_neurons) {
         rg_inhibitory_layout[j] = i;
      }
   }

   for (int i = 0; i < num_inhibitory_neurons; i++) {
      neuronTypeMap_[rg_inhibitory_layout[i]] = INH;
   }
   delete[] rg_inhibitory_layout;

   DEBUG(cout << "Done initializing neuron type map" << endl;);
}

/*
 *  Populates the starter map.
 *  Selects num_endogenously_active_neurons excitory neurons 
 *  and converts them into starter neurons.
 *
 *  @param  num_neurons number of neurons to have in the map.
 */
void DynamicLayout::initStarterMap(const int num_neurons) {
   Layout::initStarterMap(num_neurons);

   numEndogenouslyActiveNeurons_ = (BGSIZE) (m_frac_starter_neurons * num_neurons + 0.5);
   BGSIZE starters_allocated = 0;

   DEBUG(cout << "\nRandomly initializing starter map\n";);
   DEBUG(cout << "Total neurons: " << num_neurons << endl;)
   DEBUG(cout << "Starter neurons: " << numEndogenouslyActiveNeurons_ << endl;)

   // randomly set neurons as starters until we've created enough
   while (starters_allocated < numEndogenouslyActiveNeurons_) {
      // Get a random integer
      int i = static_cast<int>(rng.inRange(0, num_neurons));

      // If the neuron at that index is excitatory and a starter map
      // entry does not already exist, add an entry.
      if (neuronTypeMap_[i] == EXC && starterMap_[i] == false) {
         starterMap_[i] = true;
         starters_allocated++;
         DEBUG_MID(cout << "allocated EA neuron at random index [" << i << "]" << endl;);
      }
   }

   DEBUG(cout << "Done randomly initializing starter map\n\n";)
}
