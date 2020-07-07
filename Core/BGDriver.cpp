/*
 *  The driver for braingrid.
 *  The driver performs the following steps:
 *  1) reads parameters from an xml file (specified as the first argument)
 *  2) creates the network
 *  3) launches the simulation
 *
 *  @authors Allan Ortiz and Cory Mayberry.
 */

#include <fstream>
#include "Global.h"
#include "ParamContainer.h"

#include "IModel.h"
#include "FClassOfCategory.h"
#include "IRecorder.h"
#include "FSInput.h"
#include "Simulator.h"

// Uncomment to use visual leak detector (Visual Studios Plugin)
// #include <vld.h>


//! Cereal
#include <cereal/archives/xml.hpp>
#include <cereal/archives/binary.hpp>
#include "ConnGrowth.h"

#if defined(USE_GPU)
    #include "GPUSpikingModel.h"
#elif defined(USE_OMP)
//    #include "MultiThreadedSim.h"
#else 
    #include "SingleThreadedSpikingModel.h"
#endif

using namespace std;

// functions
bool LoadAllParameters(SimulationInfo *simInfo);
void printParams(SimulationInfo *simInfo);
bool parseCommandLine(int argc, char* argv[], SimulationInfo *simInfo);
bool createAllModelClassInstances(TiXmlDocument* simDoc, SimulationInfo *simInfo);
void printKeyStateInfo(SimulationInfo *simInfo);
void serializeSynapseInfo(SimulationInfo *simInfo, Simulator *simulator);
bool deserializeSynapseInfo(SimulationInfo *simInfo, Simulator *simulator);

/*
 *  Main for Simulator. Handles command line arguments and loads parameters
 *  from parameter file. All initial loading before running simulator in Network
 *  is here.
 *
 *  @param  argc    argument count.
 *  @param  argv    arguments.
 *  @return -1 if error, else if success.
 */
int main(int argc, char* argv[]) {
    Simulator *simulator = Simulator::getInstance();

    // Handles parsing of the command line
    if (!parseCommandLine(argc, argv)) {
        cerr << "! ERROR: failed during command line parse" << endl;
        return -1;
    }

    // Create all model instances and load parameters from a file.
    if (!LoadAllParameters()) {
        cerr << "! ERROR: failed while parsing simulation parameters." << endl;
        return -1;
    }

    // create & init simulation recorder
    simRecorder = model->getConnections()->createRecorder(simInfo);
    if (simRecorder == NULL) {
        cerr << "! ERROR: invalid state output file name extension." << endl;
        return -1;
    }

    // Create a stimulus input object
    simInfo->pInput = FSInput::get()->CreateInstance(simInfo);

    time_t start_time, end_time;
    time(&start_time);
	
    // setup simulation
    DEBUG(cerr << "Setup simulation." << endl;)
    simulator->setup();

    // Deserializes internal state from a prior run of the simulation
    if (!simInfo->memInputFileName.empty()) {
        DEBUG(cerr << "Deserializing state from file." << endl;)
        
        DEBUG(
        // Prints out internal state information before deserialization
        cout << "------------------------------Before Deserialization:------------------------------" << endl;
        printKeyStateInfo();
        )

        // Deserialization
        if(!deserializeSynapseInfo(simulator)) {
            cerr << "! ERROR: failed while deserializing objects" << endl;
            return -1;
        }

        DEBUG(
        // Prints out internal state information after deserialization
        cout << "------------------------------After Deserialization:------------------------------" << endl;
        printKeyStateInfo();
        )
    }

    // Run simulation
    simulator->simulate();

    // Terminate the stimulus input 
    if (simInfo->pInput != NULL)
    {
        simInfo->pInput->term(simInfo);
        delete simInfo->pInput;
    }

    // Writes simulation results to an output destination
    simulator->saveData(simInfo);

    // Serializes internal state for the current simulation
    if (!simInfo->memOutputFileName.empty()) {

        // Serialization
        serializeSynapseInfo(simInfo, simulator);

        DEBUG(
        // Prints out internal state information after serialization
        cout << "------------------------------After Serialization:------------------------------" << endl;
        printKeyStateInfo(simInfo);
        )

    }

    // Tell simulation to clean-up and run any post-simulation logic.
    simulator->finish(simInfo);

    // terminates the simulation recorder
    if (simInfo->simRecorder != NULL) {
        simInfo->simRecorder->term();
    }

    for(unsigned int i = 0; i < rgNormrnd.size(); ++i) {
        delete rgNormrnd[i];
    }

    rgNormrnd.clear();

    time(&end_time);
    double time_elapsed = difftime(end_time, start_time);
    double ssps = epochDuration * maxSteps / time_elapsed;
    cout << "time simulated: " << epochDuration * maxSteps << endl;
    cout << "time elapsed: " << time_elapsed << endl;
    cout << "ssps (simulation seconds / real time seconds): " << ssps << endl;
    
    delete model;
    model = NULL;

    if (simRecorder != NULL) {
        delete simRecorder;
        simRecorder = NULL;
    }

    delete simulator;
    simulator = NULL;

    return 0;
}

/*
 *  Create instances of all model classes.
 *
 *  @param  simDoc  the TiXmlDocument to read from.
 *  @param  simInfo   SimulationInfo class to read information from.
 *  @retrun true if successful, false if not
 */
bool createAllModelClassInstances(TiXmlDocument* simDoc, SimulationInfo *simInfo)
{
    TiXmlElement* parms = NULL;

    //cout << "Child:" <<  simDoc->FirstChildElement()->Value() << endl;

    if ((parms = simDoc->FirstChildElement()->FirstChildElement("ModelParams")) == NULL) {
        cerr << "Could not find <MoelParms> in simulation parameter file " << endl;
        return false;
    }

    // create neurons, synapses, connections, and layout objects specified in the description file
    IAllNeurons *neurons = NULL;
    IAllSynapses *synapses = NULL;
    Connections *conns = NULL;
    Layout *layout = NULL;
    const TiXmlNode* pNode = NULL;

    while ((pNode = parms->IterateChildren(pNode)) != NULL) {
        if (strcmp(pNode->Value(), "NeuronsParams") == 0) {
            neurons = FClassOfCategory::get()->createNeurons(pNode);
        } else if (strcmp(pNode->Value(), "SynapsesParams") == 0) {
            synapses = FClassOfCategory::get()->createSynapses(pNode);
        } else if (strcmp(pNode->Value(), "ConnectionsParams") == 0) {
            conns = FClassOfCategory::get()->createConnections(pNode);
        } else if (strcmp(pNode->Value(), "LayoutParams") == 0) {
            layout = FClassOfCategory::get()->createLayout(pNode);
        }
    }

    if (neurons == NULL){ cout << "N" << endl;}
    if (synapses == NULL){ cout << "S" << endl;}
    if (conns == NULL){ cout << "C" << endl;}
    if (layout == NULL){ cout << "L" << endl;}

    if (neurons == NULL || synapses == NULL || conns == NULL || layout == NULL) {
        cerr << "!ERROR: failed to create classes" << endl;
        return false;
    }

    // create the model
    #if defined(USE_GPU)
         simInfo->model = new GPUSpikingModel(conns, neurons, synapses, layout);
    #else
         simInfo->model = new SingleThreadedSpikingModel(conns, neurons, synapses, layout);
    #endif

    return true;
}

/*
 *  Load parameters from a file.
 *
 *  @param  simInfo   SimulationInfo class to read information from.
 *  @return true if successful, false if not
 */
bool LoadAllParameters(SimulationInfo *simInfo)
{
    DEBUG(cerr << "reading parameters from xml file" << endl;)

    TiXmlDocument simDoc(simInfo->stateInputFileName.c_str());
    if (!simDoc.LoadFile()) {
        cerr << "Failed loading simulation parameter file "
             << simInfo->stateInputFileName << ":" << "\n\t" << simDoc.ErrorDesc()
             << endl;
        cerr << " error: " << simDoc.ErrorRow() << ", " << simDoc.ErrorCol()
             << endl;
        return false;
    }

    // load simulation parameters
    if (simInfo->readParameters(&simDoc) != true) {
        return false;
    }

    // create instances of all model classes
    DEBUG(cerr << "creating instances of all classes" << endl;)
    if (createAllModelClassInstances(&simDoc, simInfo) != true) {
        return false;
    }

    // load parameters for all models
    if (FClassOfCategory::get()->readParameters(&simDoc) != true) {
        return false;
    }

    if (simInfo->stateOutputFileName.empty()) {
        cerr << "! ERROR: no stateOutputFileName is specified." << endl;
        return -1;
    }

    /*    verify that params were read correctly */
    DEBUG(printParams(simInfo);)

    return true;
}

/*
 *  Prints loaded parameters out to console.
 *
 *  @param  simInfo   SimulationInfo class to read information from.
 */
void printParams(SimulationInfo *simInfo) {
    cout << "\nPrinting simulation parameters...\n";
    simInfo->printParameters(cout);

    cout << "Model Parameters:" << endl;
    FClassOfCategory::get()->printParameters(cout);
    cout << "Done printing parameters" << endl;
}

/*
 *  Handles parsing of the command line
 *
 *  @param  argc      argument count.
 *  @param  argv      arguments.
 *  @param  simInfo   SimulationInfo class to read information from.
 *  @returns    true if successful, false otherwise.
 */
bool parseCommandLine(int argc, char* argv[], SimulationInfo *simInfo)
{
    ParamContainer cl;
    cl.initOptions(false);  // don't allow unknown parameters
    cl.setHelpString(string("The DCT growth modeling simulator\nUsage: ") + argv[0] + " ");

#if defined(USE_GPU)
    if ((cl.addParam("stateoutfile", 'o', ParamContainer::filename, "simulation state output filename") != ParamContainer::errOk)
            || (cl.addParam("stateinfile", 't', ParamContainer::filename | ParamContainer::required, "simulation state input filename") != ParamContainer::errOk)
            || (cl.addParam("deviceid", 'd', ParamContainer::regular, "CUDA device id") != ParamContainer::errOk)
            || (cl.addParam( "stiminfile", 's', ParamContainer::filename, "stimulus input file" ) != ParamContainer::errOk)
            || (cl.addParam("meminfile", 'r', ParamContainer::filename, "simulation memory image input filename") != ParamContainer::errOk)
            || (cl.addParam("memoutfile", 'w', ParamContainer::filename, "simulation memory image output filename") != ParamContainer::errOk)) {
        cerr << "Internal error creating command line parser" << endl;
        return false;
    }
#else    // !USE_GPU
    if ((cl.addParam("stateoutfile", 'o', ParamContainer::filename, "simulation state output filename") != ParamContainer::errOk)
            || (cl.addParam("stateinfile", 't', ParamContainer::filename | ParamContainer::required, "simulation state input filename") != ParamContainer::errOk)
            || (cl.addParam( "stiminfile", 's', ParamContainer::filename, "stimulus input file" ) != ParamContainer::errOk)
            || (cl.addParam("meminfile", 'r', ParamContainer::filename, "simulation memory image filename") != ParamContainer::errOk)
            || (cl.addParam("memoutfile", 'w', ParamContainer::filename, "simulation memory image output filename") != ParamContainer::errOk)) {
        cerr << "Internal error creating command line parser" << endl;
        return false;
    }
#endif  // USE_GPU

    // Parse the command line
    if (cl.parseCommandLine(argc, argv) != ParamContainer::errOk) {
        cl.dumpHelp(stderr, true, 78);
        return false;
    }

    // Get the values
    simInfo->stateOutputFileName = cl["stateoutfile"];
    simInfo->stateInputFileName = cl["stateinfile"];
    simInfo->memInputFileName = cl["meminfile"];
    simInfo->memOutputFileName = cl["memoutfile"];
    simInfo->stimulusInputFileName = cl["stiminfile"];

#if defined(USE_GPU)
    if (EOF == sscanf(cl["deviceid"].c_str(), "%d", &g_deviceId)) {
        g_deviceId = 0;
    }
#endif  // USE_GPU

    return true;
}


/*
 *  Prints key internal state information 
 *  (Used for serialization/deserialization verification)
 *
 *  @param  simInfo   SimulationInfo class to read information from.
 */
void printKeyStateInfo(SimulationInfo *simInfo)
{        
#if defined(USE_GPU)
    // Prints out SynapsesProps on the GPU
    dynamic_cast<GPUSpikingModel *>(simInfo->model)->printGPUSynapsesPropsModel();   
#else
    // Prints out SynapsesProps on the CPU
    dynamic_cast<AllSynapses *>(dynamic_cast<Model *>(simInfo->model)->m_synapses)->printSynapsesProps(); 
#endif
    // Prints out radii on the CPU (only if it is a connGrowth model) (radii is only calculated on CPU, for both CPU-based and GPU-based simulation)
    if(dynamic_cast<ConnGrowth *>(dynamic_cast<Model *>(simInfo->model)->m_conns) != nullptr) {
        dynamic_cast<ConnGrowth *>(dynamic_cast<Model *>(simInfo->model)->m_conns)->printRadii();
    }      
}

/*
 *  Serializes synapse weights, source neurons, destination neurons, 
 *  maxSynapsesPerNeuron, totalNeurons, and
 *  if running a connGrowth model, serializes radii as well 
 *
 *  @param  simInfo   SimulationInfo class to read information from.
 *  @param  simulator Simulator class to perform actions.
 */
void serializeSynapseInfo(SimulationInfo *simInfo, Simulator *simulator)
{
    // We can serialize to a variety of archive file formats. Below, comment out
    // all but the two lines that correspond to the desired format.
    ofstream memory_out (simInfo->memOutputFileName.c_str());
    cereal::XMLOutputArchive archive(memory_out);
    //ofstream memory_out (simInfo->memOutputFileName.c_str(), std::ios::binary);
    //cereal::BinaryOutputArchive archive(memory_out);

#if defined(USE_GPU)        
    // Copies GPU Synapse props data to CPU for serialization
    simulator->copyGPUSynapseToCPU(simInfo);
#endif // USE_GPU

    // Serializes synapse weights along with each synapse's source neuron and destination neuron
    archive(*(dynamic_cast<AllSynapses *>(dynamic_cast<Model *>(simInfo->model)->m_synapses)));

    // Serializes radii (only if it is a connGrowth model)
    if(dynamic_cast<ConnGrowth *>(dynamic_cast<Model *>(simInfo->model)->m_conns) != nullptr) {
        archive(*(dynamic_cast<ConnGrowth *>(dynamic_cast<Model *>(simInfo->model)->m_conns)));
    }

}

/*
 *  Deserializes synapse weights, source neurons, destination neurons,
 *  maxSynapsesPerNeuron, totalNeurons, and
 *  if running a connGrowth model and radii is in serialization file, deserializes radii as well
 *
 *  @param  simInfo   SimulationInfo class to read information from.
 *  @param  simulator Simulator class to perform actions.
 *  @returns    true if successful, false otherwise.
 */
bool deserializeSynapseInfo(SimulationInfo *simInfo, Simulator *simulator)
{
    // We can deserialize from a variety of archive file formats. Below, comment
    // out all but the line that is compatible with the desired format.
    ifstream memory_in(simInfo->memInputFileName.c_str());
    //ifstream memory_in (simInfo->memInputFileName.c_str(), std::ios::binary);

    // Checks to see if serialization file exists
    if(!memory_in) {
        cerr << "The serialization file doesn't exist" << endl;
        return false;
    }

    // We can deserialize from a variety of archive file formats. Below, comment
    // out all but the line that corresponds to the desired format.
    cereal::XMLInputArchive archive(memory_in);
    //cereal::BinaryInputArchive archive(memory_in);

    // Deserializes synapse weights along with each synapse's source neuron and destination neuron
    // Uses "try catch" to catch any cereal exception
    try {
        archive(*(dynamic_cast<AllSynapses *>(dynamic_cast<Model *>(simInfo->model)->m_synapses)));
    }
    catch(cereal::Exception e) {
        cerr << "Failed deserializing synapse weights, source neurons, and/or destination neurons." << endl;
        return false;
    }
    
    // Creates synapses from weights
    dynamic_cast<Model *>(simInfo->model)->m_conns->createSynapsesFromWeights(simInfo->totalNeurons, simInfo, dynamic_cast<Model *>(simInfo->model)->m_layout, *(dynamic_cast<Model *>(simInfo->model)->m_neurons), *(dynamic_cast<Model *>(simInfo->model)->m_synapses));

#if defined(USE_GPU)
    // Copies CPU Synapse data to GPU after deserialization, if we're doing
    // a GPU-based simulation.
    simulator->copyCPUSynapseToGPU(simInfo);
#endif // USE_GPU

    // Creates synapse index map (includes copy CPU index map to GPU)
    dynamic_cast<Model *>(simInfo->model)->m_synapses->createSynapseImap( dynamic_cast<Model *>(simInfo->model)->m_synapseIndexMap, simInfo );

#if defined(USE_GPU)
    dynamic_cast<GPUSpikingModel *>(simInfo->model)->copySynapseIndexMapHostToDevice(*(dynamic_cast<GPUSpikingModel *>(simInfo->model)->m_synapseIndexMap), simInfo->totalNeurons);
#endif // USE_GPU

    // Deserializes radii (only when running a connGrowth model and radii is in serialization file)
    if( dynamic_cast<ConnGrowth *>(dynamic_cast<Model *>(simInfo->model)->m_conns) != nullptr) {
        // Uses "try catch" to catch any cereal exception
        try {
            archive(*(dynamic_cast<ConnGrowth *>(dynamic_cast<Model *>(simInfo->model)->m_conns)));
        }
        catch(cereal::Exception e) {
            cerr << "Failed deserializing radii." << endl;
            return false;
        }
    }

    return true;

}
