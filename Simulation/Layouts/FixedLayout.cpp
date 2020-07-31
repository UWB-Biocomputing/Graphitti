#include "FixedLayout.h"
#include "ParseParamError.h"
#include "Util.h"

FixedLayout::FixedLayout() : Layout()
{
}

FixedLayout::~FixedLayout()
{
}

/*
 *  Checks the number of required parameters.
 *
 * @return true if all required parameters were successfully read, false otherwise.
 */
bool FixedLayout::checkNumParameters()
{
    return (nParams >= 1);
}

/*
 *  Prints out all parameters of the layout to ostream.
 *  @param  output  ostream to send output to.
 */
void FixedLayout::printParameters(ostream &output) const
{
    Layout::printParameters(output);

    output << "Layout parameters:" << endl;

    cout << "\tEndogenously active neuron positions: ";
    for (BGSIZE i = 0; i < num_endogenously_active_neurons; i++) {
        output << m_endogenously_active_neuron_list[i] << " ";
    }

    cout << endl;

    cout << "\tInhibitory neuron positions: ";
    for (BGSIZE i = 0; i < m_inhibitory_neuron_layout.size(); i++) {
        output << m_inhibitory_neuron_layout[i] << " ";
    }

    cout << endl;

    cout << "\tProbed neuron positions: ";
    for (BGSIZE i = 0; i < m_probed_neuron_list.size(); i++) {
        output << m_probed_neuron_list[i] << " ";
    }

    output << endl;
}

/*
 *  Creates a randomly ordered distribution with the specified numbers of neuron types.
 *  @param  num_neurons number of the neurons to have in the type map.
 *  @return a flat vector (to map to 2-d [x,y] = [i % m_width, i / m_width])
 */
void FixedLayout::generateNeuronTypeMap(int num_neurons)
{
    Layout::generateNeuronTypeMap(num_neurons);

    int num_inhibitory_neurons = m_inhibitory_neuron_layout.size();
    int num_excititory_neurons = num_neurons - num_inhibitory_neurons;

    DEBUG(cout << "Total neurons: " << num_neurons << endl;)
    DEBUG(cout << "Inhibitory Neurons: " << num_inhibitory_neurons << endl;)
    DEBUG(cout << "Excitatory Neurons: " << num_excititory_neurons << endl;)

    for (int i = 0; i < num_inhibitory_neurons; i++) {
        assert(m_inhibitory_neuron_layout.at(i) < num_neurons);
        neuron_type_map[m_inhibitory_neuron_layout.at(i)] = INH;
    }

    DEBUG(cout << "Done initializing neuron type map" << endl;);
}

/*
 *  Populates the starter map.
 *  Selects \e numStarter excitory neurons and converts them into starter neurons.
 *  @param  num_neurons number of neurons to have in the map.
 */
void FixedLayout::initStarterMap(const int num_neurons)
{
   Layout::initStarterMap(num_neurons);

    for (BGSIZE i = 0; i < num_endogenously_active_neurons; i++) {
        assert(m_endogenously_active_neuron_list.at(i) < num_neurons);
        starter_map[m_endogenously_active_neuron_list.at(i)] = true;
    }
}
