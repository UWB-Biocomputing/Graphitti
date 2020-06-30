/*
 * @file Simulator.cpp
 *
 * @author Derek McLean
 *
 * @brief Base class for model-independent simulators targeting different
 * platforms.
 */

#include "Simulator.h"
#include "ParseParamError.h"                                            // added from siminfo

Simulator *Simulator::instance = nullptr; 

// Get Instance method that acts as a constructor, returns the instance of the singleton object
static Simulator *Simulator::getInstance() {
  if (instance == nullptr) {
    instance = Simulator;    
  }
  return instance;
};

/*
 *  Constructor
 */
Simulator::Simulator() 
{
  g_simulationStep = 0;
}

/*
 * Destructor.
 */
Simulator::~Simulator()
{
  freeResources();
}

/*
 *  Initialize and prepare network for simulation.
 *
 *  @param  sim_info    parameters for the simulation.
 */
void Simulator::setup()
{
#ifdef PERFORMANCE_METRICS
  // Start overall simulation timer
  cerr << "Starting main timer... ";
  t_host_initialization_layout = 0.0;
  t_host_initialization_connections = 0.0;
  t_host_advance = 0.0;
  t_host_adjustSynapses = 0.0;
  timer.start();
  cerr << "done." << endl;
#endif

  DEBUG(cerr << "Initializing models in network... ";)
  model->setupSim();
  DEBUG(cerr << "\ndone init models." << endl;)

  // init stimulus input object
  if (pInput != NULL) {
    cout << "Initializing input." << endl;
    pInput->init();
  }
}

/*
 *  Begin terminating the simulator.
 *
 *  @param  sim_info    parameters for the simulation.
 */
void Simulator::finish()
{
  // Terminate the simulator
  model->cleanupSim(); // Can #term be removed w/ the new model architecture?  // =>ISIMULATION
}

// *********************    added from siminfo      ***********************


/*
 *  Attempts to read parameters from a XML file.
 *
 *  @param  simDoc  the TiXmlDocument to read from.
 *  @return true if successful, false otherwise.
 */
bool Simulator::readParameters(TiXmlDocument* simDoc)
{
    TiXmlElement* parms = NULL;

    if ((parms = simDoc->FirstChildElement()->FirstChildElement("SimInfoParams")) == NULL) {
        cerr << "Could not find <SimInfoParams> in simulation parameter file " << endl;
        return false;
    }

    try {
         parms->Accept(this);
    } catch (ParseParamError &error) {
        error.print(cerr);
        cerr << endl;
        return false;
    }

    // check to see if all required parameters were successfully read
    if (checkNumParameters() != true) {
        cerr << "Some parameters are missing in <SimInfoParams> in simulation parameter file " << endl;
        return false;
    }

    return true;
}

/*
 *  Handles loading of parameters using tinyxml from the parameter file.
 *
 *  @param  element TiXmlElement to examine.
 *  @param  firstAttribute  ***NOT USED***.
 *  @return true if method finishes without errors.
 */
bool Simulator::VisitEnter(const TiXmlElement& element, const TiXmlAttribute* firstAttribute)
//TODO: firstAttribute does not seem to be used! Delete?
{
    static string parentNode = "";
    if (element.ValueStr().compare("SimInfoParams") == 0) {
        return true;
    }

    if (element.ValueStr().compare("PoolSize")      == 0   ||
        element.ValueStr().compare("SimParams")     == 0   ||
	      element.ValueStr().compare("SimConfig")     == 0   ||
	      element.ValueStr().compare("Seed")          == 0   ||
	      element.ValueStr().compare("OutputParams")  == 0    ) {
	        nParams++;                                                                               // where is first value of nparams being sourced from? declared but cant find initial value. 

          return true;
          }

    if (element.Parent()->ValueStr().compare("PoolSize") == 0) {
      if(element.ValueStr().compare("x") == 0){
        width = atoi(element.GetText());
        }
      else if(element.ValueStr().compare("y") == 0){
        height = atoi(element.GetText());
        }

      if(width != 0 && height != 0){
        totalNeurons = width * height;
        }
      return true;
      }

    if (element.Parent()->ValueStr().compare("SimParams") == 0) {

    if(element.ValueStr().compare("Tsim") == 0){
        epochDuration = atof(element.GetText());
    }
    else if(element.ValueStr().compare("numSims") == 0){
        maxSteps = atof(element.GetText());
    }

          if (epochDuration < 0 || maxSteps < 0) {
              throw ParseParamError("SimParams", "Invalid negative SimParams value.");
          }

          return true;
      }

    if (element.Parent()->ValueStr().compare("SimConfig") == 0) {
/*
        if (element.QueryIntAttribute("maxFiringRate", &maxFiringRate) != TIXML_SUCCESS) {
            throw ParseParamError("SimConfig maxFiringRate", "SimConfig missing maxFiringRate value in XML.");
        }
        if (element.QueryIntAttribute("maxSynapsesPerNeuron", &maxSynapsesPerNeuron) != TIXML_SUCCESS) {
            throw ParseParamError("SimConfig maxSynapsesPerNeuron", "SimConfig missing maxSynapsesPerNeuron value in XML.");
        }
        nParams++;
*/
	if(element.ValueStr().compare("maxFiringRate") == 0){
	    maxFiringRate = atoi(element.GetText());
	}
	else if(element.ValueStr().compare("maxSynapsesPerNeuron") == 0){
	    maxSynapsesPerNeuron = atoi(element.GetText());
	}

        if (maxFiringRate < 0 || maxSynapsesPerNeuron < 0) {
            throw ParseParamError("SimConfig", "Invalid negative SimConfig value.");
        }

        return true;
    }

    if (element.Parent()->ValueStr().compare("Seed") == 0) {
/*
        if (element.QueryValueAttribute("value", &seed) != TIXML_SUCCESS) {
            throw ParseParamError("Seed value", "Seed missing value in XML.");
        }
        nParams++;
*/
	if(element.ValueStr().compare("value") == 0){
	    seed = atoi(element.GetText());
	}
        return true;
    }

    if (element.Parent()->ValueStr().compare("OutputParams") == 0) {
        // file name specified in commond line is higher priority

        if (stateOutputFileName.empty()) {
/*
            if (element.QueryValueAttribute("stateOutputFileName", &stateOutputFileName) != TIXML_SUCCESS) {
                throw ParseParamError("OutputParams stateOutputFileName", "OutputParams missing stateOutputFileName value in XML.");
            }
*/
	    if(element.ValueStr().compare("stateOutputFileName") == 0){
		stateOutputFileName = element.GetText();
	    }
        }

//      nParams++;
        return true;
    }

    return false;
}

/*
 *  Prints out loaded parameters to ostream.
 *
 *  @param  output  ostream to send output to.
 */
void Simulator::printParameters(ostream &output) const
{
    cout << "poolsize x:" << width << " y:" << height
         //z dimmension is for future expansion and not currently supported
         //<< " z:" <<
         << endl;
    cout << "Simulation Parameters:\n";
    cout << "\tTime between growth updates (in seconds): " << epochDuration << endl;
    cout << "\tNumber of simulations to run: " << maxSteps << endl;
}



// ********************** finish added from siminfo *********************
/**
 * Copy GPU Synapse data to CPU.
 *
 *  @param  sim_info    parameters for the simulation.
 */
void Simulator::copyGPUSynapseToCPU() {
  model->copyGPUSynapseToCPUModel(sim_info); 
}

 /**
 * Copy CPU Synapse data to GPU.
 *
 *  @param  sim_info    parameters for the simulation.
 */
void Simulator::copyCPUSynapseToGPU() {
  model->copyCPUSynapseToGPUModel(); 
}		

/*
 * Resets all of the maps.
 * Releases and re-allocates memory for each map, clearing them as necessary.
 *
 *  @param  sim_info    parameters for the simulation.
 */
void Simulator::reset()
{
  DEBUG(cout << "\nEntering Simulator::reset()" << endl;)

    // Terminate the simulator
    model->cleanupSim();

  // Clean up objects
  freeResources();

  // Reset global simulation Step to 0
  g_simulationStep = 0;

  // Initialize and prepare network for simulation
  model->setupSim();

  DEBUG(cout << "\nExiting Simulator::reset()" << endl;)
    }

/*
 *  Clean up objects.
 */
void Simulator::freeResources()
{
}

/*
 * Run simulation
 *
 *  @param  sim_info    parameters for the simulation.
 */
void Simulator::simulate()
{
  // Main simulation loop - execute maxGrowthSteps
  for (int currentStep = 1; currentStep <= maxSteps; currentStep++) {

    DEBUG(cout << endl << endl;)
      DEBUG(cout << "Performing simulation number " << currentStep << endl;)
      DEBUG(cout << "Begin network state:" << endl;)

      // Init SimulationInfo parameters
      currentStep = currentStep;

#ifdef PERFORMANCE_METRICS
      // Start timer for advance
      short_timer.start();
#endif
    // Advance simulation to next growth cycle
    advanceUntilGrowth(currentStep);
#ifdef PERFORMANCE_METRICS
    // Time to advance
    t_host_advance += short_timer.lap() / 1000000.0;
#endif

    DEBUG(cout << endl << endl;)
      DEBUG(
            cout << "Done with simulation cycle, beginning growth update "
	    << currentStep << endl;
	    )

      // Update the neuron network
#ifdef PERFORMANCE_METRICS
      // Start timer for connection update
      short_timer.start();
#endif
    model->updateConnections();
      
    model->updateHistory();

#ifdef PERFORMANCE_METRICS
    // Times converted from microseconds to seconds
    // Time to update synapses
    t_host_adjustSynapses += short_timer.lap() / 1000000.0;
    // Time since start of simulation
    double total_time = timer.lap() / 1000000.0;

    cout << "\ntotal_time: " << total_time << " seconds" << endl;
    printPerformanceMetrics(total_time, currentStep);
    cout << endl;
#endif
  }
}

/*
 * Helper for #simulate().
 * Advance simulation until it's ready for the next growth cycle. This should simulate all neuron and
 * synapse activity for one epoch.
 *
 *  @param currentStep the current epoch in which the network is being simulated.
 *  @param  sim_info    parameters for the simulation.
 */
void Simulator::advanceUntilGrowth(const int currentStep)
{
  uint64_t count = 0;
  // Compute step number at end of this simulation epoch
  uint64_t endStep = g_simulationStep
    + static_cast<uint64_t>(epochDuration / deltaT);

  DEBUG_MID(model->logSimStep();) // Generic model debug call

    while (g_simulationStep < endStep) {
      DEBUG_LOW(
		// Output status once every 10,000 steps
		if (count % 10000 == 0)
		  {
		    cout << currentStep << "/" << maxSteps
			 << " simulating time: "
			 << g_simulationStep * deltaT << endl;
		    count = 0;
		  }
		count++;
		)

        // input stimulus
        if (pInput != NULL)
	  pInput->inputStimulus();

      // Advance the Network one time step
      model->advance();

      g_simulationStep++;
    }
}

/*
 * Writes simulation results to an output destination.
 * 
 *  @param  sim_info    parameters for the simulation. 
 */
void Simulator::saveData() const
{
  model->saveData();
}

//! Width of neuron map (assumes square)
int Simulator::getWidth() {return width;}

//! Height of neuron map
int Simulator::getHeight() {return height;}

//! Count of neurons in the simulation
int Simulator::getTotalNeurons() {return totalNeurons;}

//! Current simulation step
int Simulator::getCurrentStep() {return currentStep;}

//! Maximum number of simulation steps
int Simulator::getMaxSteps() {return maxSteps;}

//! The length of each step in simulation time
BGFLOAT Simulator::getEpochDuration() {return epochDuration;}

//! Maximum firing rate. **Only used by GPU simulation.**
int Simulator::getMaxFiringRate() {return maxFiringRate;}

//! Maximum number of synapses per neuron. **Only used by GPU simulation.**
int Simulator::getMaxSynapsesPerNeuron() {return maxSynapsesPerNeuron;}

//! Time elapsed between the beginning and end of the simulation step
BGFLOAT Simulator::getDeltaT() {return deltaT;}

//! The neuron type map (INH, EXC).
neuronType* Simulator::getRgNeuronTypeMap() {return rgNeuronTypeMap;}

//! The starter existence map (T/F).
bool* Simulator::getRgEndogenouslyActiveNeuronMap() {return rgEndogenouslyActiveNeuronMap;}

//! growth variable (m_targetRate / m_epsilon) TODO: more detail here
BGFLOAT Simulator::getMaxRate() {return maxRate;}

//! List of summation points (either host or device memory)
BGFLOAT* Simulator::getPSummationMap() {return pSummationMap;}

//! Seed used for the simulation random SINGLE THREADED
long Simulator::getSeed() {return seed;}

//! File name of the simulation results.
string Simulator::getStateOutputFileName() {return stateOutputFileName;}

//! File name of the parameter description file.
string Simulator::getStateInputFileName() {return stateInputFileName;}

//! File name of the memory dump output file.
string Simulator::getMemOutputFileName() {return memOutputFileName;}

//! File name of the memory dump input file.
string Simulator::getMemInputFileName() {return memInputFileName;}

//! File name of the stimulus input file.
string Simulator::getStimulusInputFileName() {return stimulusInputFileName;}

//! Neural Network Model interface.
IModel* Simulator::getModel() {return model;}

//! Recorder object.
IRecorder* Simulator::getSimRecorder() {return simRecorder;}

//! Stimulus input object.
ISInput* Simulator::getPInput() {return pInput;}

/**
  * Timer for measuring performance of an epoch.
  */
Timer Simulator::getTimer() {return timer;}

/**
  * Timer for measuring performance of connection update.
  */
Timer Simulator::getShort_timer() {return short_timer;}

