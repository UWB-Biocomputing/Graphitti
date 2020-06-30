/**
 * @file Simulator.h
 *
 * @brief Platform independent base class for the Brain Grid simulator. 
 *
 * @ingroup Core
 *
 * @authors contributing Derek McLean & Sean Blackbourn       // ToDo: Authorship timeline in main docs. 
 *
 */
#pragma once                                                                  // modern ifendif

// todo: revisit big decisions here for high level architecture 
#include "Global.h"
#include "IModel.h"  // model owns connections, layout
#include "ISInput.h"
// class IRecorder;                                                        // simulator.h. IRecorder is not in simulator. 

#ifdef PERFORMANCE_METRICS                                                  // added from siminfo
// Home-brewed performance measurement  *doesnt affect runtime itself. *also collects performance on GPU *warner smidt paper with details "profiling braingrid"
#include "Timer.h"
#endif

/**
 * @class Simulator Simulator.h "Simulator.h"
 *
 *
 * This class should be extended when developing the simulator for a specific platform.
 *
 * \latexonly  \subsubsection*{Credits} \endlatexonly
 * \htmlonly   <h3>Credits</h3> \endhtmlonly
 *
 * Some models in this simulator is a rewrite of CSIM (2006) and other
 * work (Stiber and Kawasaki (2007?))
 */
class Simulator : public TiXmlVisitor                                       // added from siminfo
{
    public:
        // Get Instance method that acts as a constructor, returns the instance of the singleton object
        static Simulator *getInstance();

        // todo: one accessor get current timestep 
        // todo: inline int getter() check if inline keyword has significance

        //! Width of neuron map (assumes square)
        int getWidth();

	//! Height of neuron map
        int getHeight();

	//! Count of neurons in the simulation
        int getTotalNeurons();

	//! Current simulation step
        int getCurrentStep();

	//! Maximum number of simulation steps
        int getMaxSteps();

	//! The length of each step in simulation time
        BGFLOAT getEpochDuration();

	//! Maximum firing rate. **Only used by GPU simulation.**
        int getMaxFiringRate();

	//! Maximum number of synapses per neuron. **Only used by GPU simulation.**
        int getMaxSynapsesPerNeuron();

	//! Time elapsed between the beginning and end of the simulation step
        BGFLOAT getDeltaT();

	//! The neuron type map (INH, EXC).
        neuronType* getRgNeuronTypeMap();

	//! The starter existence map (T/F).
        bool* getRgEndogenouslyActiveNeuronMap();

	//! growth variable (m_targetRate / m_epsilon) TODO: more detail here
        BGFLOAT getMaxRate();

	//! List of summation points (either host or device memory)
        BGFLOAT* getPSummationMap();

	//! Seed used for the simulation random SINGLE THREADED
        long getSeed();

        //! File name of the simulation results.
        string getStateOutputFileName();

        //! File name of the parameter description file.
        string getStateInputFileName();

        //! File name of the memory dump output file.
        string getMemOutputFileName();

        //! File name of the memory dump input file.
        string getMemInputFileName();

        //! File name of the stimulus input file.
        string getStimulusInputFileName() const;    // todo: make all const. 

        //! Neural Network Model interface.
        IModel* getModel();

        //! Recorder object.
        IRecorder* getSimRecorder();

        //! Stimulus input object.
        ISInput* getPInput();           // <--- do we need to return these? do these need to be accessible outside. figure out something else to return beside ptr. maybe safely return const reference? todo. 

        // ***** big questions: 
        // is this something other classes should be modifying? 
        // is returning a copy helpful? if modified by other classes... 
        // todo: look@ how other classes are using performance metrics. 
        // do other classes make their own timers? was this to output timing? 
        // should be read only
    #ifdef PERFORMANCE_METRICS
        /**
         * Timer for measuring performance of an epoch.
         */
        Timer getTimer();  //returns copy of internal timer owned by simulator. 
        /**
         * Timer for measuring performance of connection update.
         */
        Timer getShort_timer();
    #endif




        /** Destructor */
        virtual ~Simulator();

        /**
         * Setup simulation.
         *
         *  @param  sim_info    parameters for the simulation.
         */
        void setup(SimulationInfo *sim_info);

        /**
         * Cleanup after simulation.
         *
         *  @param  sim_info    parameters for the simulation.
         */
        void finish(SimulationInfo *sim_info);

        

        /**                                                         // added from siminfo
         *  Attempts to read parameters from a XML file.
         *
         *  @param  simDoc  the TiXmlDocument to read from.
         *  @return true if successful, false otherwise.
         */
        bool readParameters(TiXmlDocument* simDoc);   

        /**                                                       // added from siminfo
         *  Prints out loaded parameters to ostream.
         *
         *  @param  output  ostream to send output to.
         */
        void printParameters(ostream &output) const;      

        /**
         * Copy GPU Synapse data to CPU.
         *
         *  @param  sim_info    parameters for the simulation.
         */
        void copyGPUSynapseToCPU(SimulationInfo *sim_info);
        
        /**
         * Copy CPU Synapse data to GPU.
         *
         *  @param  sim_info    parameters for the simulation.
         */
        void copyCPUSynapseToGPU(SimulationInfo *sim_info);

        /** 
         * Reset simulation objects.
         *
         *  @param  sim_info    parameters for the simulation.
         */
        void reset(SimulationInfo *sim_info);

        /**
         * Performs the simulation.
         *
         *  @param  sim_info    parameters for the simulation.
         */
        void simulate();

        /**
         * Advance simulation to next growth cycle. Helper for #simulate().
         *
         *  @param currentStep the current epoch in which the network is being simulated.
         *  @param  sim_info    parameters for the simulation.
         */
        void advanceUntilGrowth(const int currentStep, SimulationInfo *sim_info);

        /**
         * Writes simulation results to an output destination.
         *
         *  @param  sim_info    parameters for the simulation.
         */
        void saveData(SimulationInfo *sim_info) const;

    protected:                                                                  // added from siminfo
        using TiXmlVisitor::VisitEnter;

        /*
         *  Handles loading of parameters using tinyxml from the parameter file.
         *
         *  @param  element TiXmlElement to examine.
         *  @param  firstAttribute  ***NOT USED***.
         *  @return true if method finishes without errors.
         */
        virtual bool VisitEnter(const TiXmlElement& element, const TiXmlAttribute* firstAttribute);

    private:
        /**
        *  Constructor
        */
	Simulator();

        // pointer to instance 
        static Simulator *instance;
        
        /**
         * Frees dynamically allocated memory associated with the maps.
         */
        void freeResources();
        
        //! Width of neuron map (assumes square)
	int width;

	//! Height of neuron map
	int height;

	//! Count of neurons in the simulation
	int totalNeurons;

	//! Current simulation step
	int currentStep;

	//! Maximum number of simulation steps
	int maxSteps; // TODO: delete

	//! The length of each step in simulation time
	BGFLOAT epochDuration; // Epoch duration !!!!!!!!

	//! Maximum firing rate. **Only used by GPU simulation.**
	int maxFiringRate;

	//! Maximum number of synapses per neuron. **Only used by GPU simulation.**
	int maxSynapsesPerNeuron;

	//! Time elapsed between the beginning and end of the simulation step
        // did so could change between float and double. purely investigative type. gives more confidence. 
	BGFLOAT deltaT; // Inner Simulation Step Duration !!!!!!!!

	//! The neuron type map (INH, EXC).
	neuronType* rgNeuronTypeMap;

	//! The starter existence map (T/F).
	bool* rgEndogenouslyActiveNeuronMap;

	//! growth variable (m_targetRate / m_epsilon) TODO: more detail here
	BGFLOAT maxRate;

	//! List of summation points (either host or device memory)
	BGFLOAT* pSummationMap;

	//! Seed used for the simulation random SINGLE THREADED
	long seed;

        //! File name of the simulation results.
        string stateOutputFileName;

        //! File name of the parameter description file.
        string stateInputFileName;

        //! File name of the memory dump output file.
        string memOutputFileName;

        //! File name of the memory dump input file.
        string memInputFileName;

        //! File name of the stimulus input file.
        string stimulusInputFileName;

        //! Neural Network Model interface.
        IModel* model;

        //! Recorder object.
        IRecorder* simRecorder;

        //! Stimulus input object.
        ISInput* pInput;

    #ifdef PERFORMANCE_METRICS
        /**
         * Timer for measuring performance of an epoch.
         */
        Timer timer;

        /**
         * Timer for measuring performance of connection update.
         */
        Timer short_timer;

    #endif


};



 