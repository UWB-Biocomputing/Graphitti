# Recorders

In the Graphitti C++ project, the Recorder subsystem is a critical component for saving simulation data. There are two primary types of recorders: XML and HDF5.

## Hierarchical Data Format (HDF5)

HDF5 is the standard file type for recording data. It's a binary format that is platform-independent, meaning floating-point numbers are not affected by the platform. The file is self-documenting, containing a list of datasets, dimensions, types, and more. Some additional information is available in the lab directory. The HDF5 format is also integrated with MATLAB, enabling the loading of chunks of data, which is useful for our purposes.

## XML

XML is a text-based format. It has advantages for smaller datasets due to its readability but can be challenging to handle with large volumes of data. Custom tools have been developed for MATLAB to facilitate the use of XML.

## How the Redesigned Recorder Works (Q1 2024)

The redesigned Recorder subsystem (Q1 2024 implementation) aims to accommodate a flexible and simplified data recording process for simulations.

### Supported Data Format

In Graphitti, the simulation data can be saved in either an XML or an HDF5 file format by specifying the `XmlRecorder` or `HDF5Recorder` in the input configuration. The main differences in how `XmlRecorder` and `HDF5Recorder` record dynamic simulation data are:

- **XmlRecorder:** Captures all data in vectors during the simulation and writes them to an XML file once the simulation is complete.
- **HDF5Recorder:** Writes data directly to the HDF5 library routines during the simulation, eliminating the need to store the entire dataset in memory at once.

### Supported Data Types

The Recorder system is capable of capturing a wide range of data types using `std::variant` from C++17. The currently supported data types include:

- `unit64_t`
- `BGFLOAT`
- `int`
- `bool`

Extending the coverage for new data types involves a two-step process:

1. Expand the `variant` structure data type list (add a new argument to the list of supported data types).
2. Modify how values are retrieved from the `variant` for data output purposes in the Recorder.

### Supported Data Structures

The current Recorder system supports 1-D variables. The recorded variables must have a base interface `RecordableBase`. Current recordable data structures include:

- `EventBuffer`
- `VectorMatrix`
- Standard library `Vector` (substitutable by `RecordableVector`)

### Updated Type

The Recorder subsystem supports two types of variables: `DYNAMIC` and `CONSTANT`. These two types are updated at different frequencies:

- `CONSTANT`: Variable value doesn’t change during the simulation epoch. The values can be capture after simulation finishing
- `DYNAMIC`: Value of dynamic variables is updated in each epoch.

### How to Record Data

Recording data with the Recorder subsystem involves several straightforward steps. Below is a general guide:

#### Step 1: Variable Registration in Variable Owner Class

First, register the variables you intend to record. Each simulation component, especially the variable owner class responsible for its variables, registers for tracking and recording by informing the Recorder of the variable's details. This includes its name, memory address, and update frequency. This is achieved by calling the `registerVariable` method in the Recorder. Note that the registered variable must be a subclass of the `RecordableBase` interface.

```cpp
recorder.registerVariable("variableName", variableAddress, updateType);
```

#### Step 2: Capturing Data and Saving Data in the Recorder

As long as the variable owner class has registered the variable of interest, the Recorder automatically adds each registered variable to the variable table, captures the data, and then outputs them to the output file during the simulation, ensuring that the simulation data is captured at the appropriate moments.

-------------------------------------------------------------------------------------------------
## Legacy

Previously, the focus was not on analyzing individual spikes but on breaking time into bins of 10ms and counting the number of spikes in each bin. This approach was detailed in a 2014 paper. The comment in the code has not been updated since. In the HDF5Recorder.cpp file, there are specific names for datasets:

```cpp
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
```

init must be called somewhere else.
this is a wrapper around file. must have been done before initdefault values . blank!
```cpp
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
```
todo: investigate blank initdefaultvalues
```cpp
/*
 * Init history matrices with default values
 */
void Hdf5Recorder::initDefaultValues()
{
}
```

Recorders need to record data during the simulation, but the data depends on which model is being simulated.

**Growth simulation**: Every neuron has a radius that grows or shrinks.

**STDP simulation**: Simulates synapse weight changes during spike times.

Depending on the model (growth or STDP), they will generate different types of data because they do different things.

Can't disentangle this structure from the structure of synapses and neurons.

**ToDo**: In param file, put which type of sim so that we can select which thing to record.

We record from neurons. For STDP, we need to record from synapses.

Going to be unavoidable interconnectivity between recorders and synapses, neurons. Sort of like a mechanism between synapses and neurons.

**ToDo**: Why does HDF5 growth recorder redo `init`? Couldn't it just inherit it?

`Hdf5Recorder` is set up to be a concrete class.

**ToDo**: Why?

**ToDo**: Structural issues about getting information from the connections class.

`radiiHistory`/`rateHistory` are member variables of `Hdf5GrowthRecorder`.

Could be expensive to use getters here.








