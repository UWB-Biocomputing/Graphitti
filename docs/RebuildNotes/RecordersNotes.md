
# Recorders
There are two different types of recorders: XML and HDF5 file types



Heirarchical data format (HDF5) standard data file type for standard data.

binary format but platform independent. ex. floating pt numbers arent affected.
self documenting. in file there is a list of data sets, dimensions, types etc.

some in the lab directory. ToDo: H5 info is a document on raiju to tell us about it.

also integrated with matlab. can load chunks of data. useful for us.

XML just text. advantages with smaller datasets. disadvantage is at sheer volume of data.

set of custom stuff for matlab to allow for

advantages of xml is testing. small sets are easier to read.



legacy: wasnt at point where they wanted to analyze individual spikes.
broke time into bins. 10ms. count of number of spikes in each 10 ms bin. 2014 paper - analysis based on this
todo: bring 2014 paper details into
look at radii.

comment hasnt been updated since
todo: update to reflect what actually does. more than radii.
    /**
     * Init radii and rates history matrices with default values
     */

     recorders record info from all probed neurons
all neurons probed. all spikes from all neurons are now recorded.

easier to record from everything- didnt slow down. look at subsets.

todo: radii and rates history - change wording to reflect current functipnality.

in hdf5recorder.cpp

// someo of this is specific to the hdf5 recorder stuff
// hdf5 dataset name
const H5std_string  nameBurstHist("burstinessHist_");
const H5std_string  nameSpikesHist("spikesHistory_");

const H5std_string  nameXloc("xloc");
const H5std_string  nameYloc("yloc");
const H5std_string  nameNeuronTypes("neuronTypes");
const H5std_string  nameNeuronThresh("neuronThresh");
const H5std_string  nameStarterNeurons("starterNeurons");
const H5std_string  nameTsim("Tsim");
const H5std_string  nameSimulationEndTime("simulationEndTime");

const H5std_string  nameSpikesProbedNeurons("spikesProbedNeurons");
const H5std_string  nameAttrPNUnit("attrPNUint");
const H5std_string  nameProbedNeurons("probedNeurons");


init must be called somewhere else.
this is a wrapper around file. must have been done before initdefault values . blank!
/*
 * Initialize data
 * Create a new hdf5 file with default properties.
 *
 * @param[in] stateOutputFileName	File name to save histories
 */
void Hdf5Recorder::init(const string& stateOutputFileName)
{
    try
    {
        // create a new file using the default property lists
        stateOut_ = new H5File( stateOutputFileName, H5F_ACC_TRUNC );

        initDataSet();
    }

    // catch failure caused by the H5File operations
    catch( FileIException error )
    {
        error.printErrorStack();
        return;
    }

    // catch failure caused by the DataSet operations
    catch( DataSetIException error )
    {
        error.printErrorStack();
        return;
    }

    // catch failure caused by the DataSpace operations
    catch( DataSpaceIException error )
    {
        error.printErrorStack();
        return;
    }

    // catch failure caused by the DataType operations
    catch( DataTypeIException error )
    {
        error.printErrorStack();
        return;
    }
}

todo: investigate blank initdefaultvalues

/*
 * Init history matrices with default values
 */
void Hdf5Recorder::initDefaultValues()
{
}

recorders need to record data during the simulation. but data depends on which model is being simulation.

Growth simulation: every neuron has a radius that grows or shrinks

stdp simulation: simulates synapse weight changes during spike times.

depending on model growth or stdp, they will generate different types of data because they do different things.

cant disentangle this structure from structure of synapses and neurons.

todo: in param file, put which type of sim so that we can select which thing to record.

we record from neurons. for stdp we need to record from synapses.

going to be unavoitable interconnectivity between recorders and synapses, neurons.
sort of like mechanism between synapses and neurons

todo: why does hdf5 growth recorder redo init? couldnt it just inherit it?

hdf5recorder is setup to be concrete class. todo: why?

todo: structural issues about getting information from connections class.

radiihistory/ratehistory are member variables of hdf5 growthrecorder.

could be expensive to use getters here.








