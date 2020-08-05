/*
 *  Setup the internal structure of the class.
 *  Allocate memories to store all layout state.
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 */
void Layout::setupLayout(const SimulationInfo *sim_info)
{
    int num_neurons = sim_info->totalNeurons;

// alloc memory
    xloc = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons);
    yloc = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons);
    dist2 = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons);
    dist = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons);

    // Initialize neuron locations memory, grab global info
    initNeuronsLocs(sim_info);

    // calculate the distance between neurons
    // precomputing. also initialization
    // given every neuron has xy location, precomputing dist between each pair of neurons.
    // todo: pull out distance calc method. 
for (int n = 0; n < num_neurons - 1; n++)
    {
        for (int n2 = n + 1; n2 < num_neurons; n2++)
        {
            // distance^2 between two points in point-slope form
            (*dist2)(n, n2) = ((*xloc)[n] - (*xloc)[n2]) * ((*xloc)[n] - (*xloc)[n2]) +
                ((*yloc)[n] - (*yloc)[n2]) * ((*yloc)[n] - (*yloc)[n2]);

            // both points are equidistant from each other
            (*dist2)(n2, n) = (*dist2)(n, n2);
        }
    }

    // take the square root to get actual distance (Pythagoras was right!)
    // (The CompleteMatrix class makes this assignment look so easy...)
    // ugliness like this because of how things were alloc. doesnt need to have this.
    (*dist) = sqrt((*dist2));

// more alloc of internal mem.
    neuron_type_map = new neuronType[num_neurons];
    starter_map = new bool[num_neurons];

    // todo: takeaway: no sequential dependency here
}



MODEL CREATE ALL NEURONS what does it do?
telling layout to do things.
todo: we want to make sure we are creating things only once.
has data structure to track type of each neuron. in generateneurontypemap
todo: why is init separate from alloc? setup does alloc/init. but some things are init separately.
no reason why there needs to be

/* todo: this method should not be separate from FixedLayout::generateNeuronTypeMap
 *  Creates a neurons type map.
 *
 *  @param  num_neurons number of the neurons to have in the type map.
 */
void Layout::generateNeuronTypeMap(int num_neurons)
{
    DEBUG(cout << "\nInitializing neuron type map"<< endl;);

    for (int i = 0; i < num_neurons; i++) {
        neuron_type_map[i] = EXC;
    }

which neurons are inhibitory is

todo: this couldve been done in setupSim.
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

all setup is doing: allocating and initializing it.
todo: is there anything tyhat needs to be done in particular order? so far, no.

there is no reason why setupSim cant do all of these things.

// this method is like neuron type map but identifies which are endogenously active.
/*
 *  Populates the starter map.
 *  Selects num_endogenously_active_neurons excitory neurons and converts them into starter neurons.
 *
 *  @param  num_neurons number of neurons to have in the map.
 */
void Layout::initStarterMap(const int num_neurons)
{
    // todo: where do some get set true in another place? find
    for (int i = 0; i < num_neurons; i++) {
        starter_map[i] = false;
    }
}

// todo: combine all setup and init methods into just setup method in layout class.

dynamic layout has randdomly selecting which are endogenous neurons.

subclass is initializing stuff and fixedlayout is setting some true.

dynamic layout changes which params are relavant.



7/29 

Holds object but necessary to create object in other place 



