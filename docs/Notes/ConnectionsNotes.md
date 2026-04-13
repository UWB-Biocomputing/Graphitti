synapses and neurons have to talk to eachother. connections facilitates this.

synapses will be inside connections class.

todo: tbd whether synapses will be a member variable or ??

layout will manage and own neurons
connections will manage synapses

layouts connections and recorders will be on same level.
Model owns reorders connections layouts


todo: connGrowth.cpp is messy and could be cleaned up.
actual computation occurs in connGrowth

todo: ConnGrowth shouldnt have own copies of these. should come from their home.
Share from sim info instead of make copies

void ConnGrowth::setupConnections
    W = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons, 0);
    radii = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons, m_growth.startRadius);
    rates = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons, 0);
    delta = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons);
    area = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons, 0);
    outgrowth = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons);
    deltaR = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons);


using stuff to create and destroy things.

properties of neurons needed to perform certain actions in connections

computation happening in one place. copying data into other place.
// even though it seems more invasive, makes more sense to not have two copies. just write it to home.

connGrowth assumes there is no synapses at startup time
sequential dependency. synapses->addSynapses is actually creating synapses. in ConnStatic::setupConnections.
stuff needs to happen before
all the synapse stuff needs to be alloc and init before setupConnections can happen. in Model::setupSim m_synapses->createSynapseImap

setupConnections needs to happen after some stuff.


