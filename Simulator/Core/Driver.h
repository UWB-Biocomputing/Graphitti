/**
 * @file Driver.h
 * 
 * @ingroup Simulator/Core
 *
 * @brief Orchestrates most functionality in the simulation.
 * 
 *  The setupSimulation performs the following steps:
 *  1) Instantiates Simulator object
 *  2) Parses command line to get configuration file and additional information if provided
 *  3) Loads global Simulator parameters from configuration file
 *  4) Instantiates all simulator objects (Layout, Connections, Synapases, Vertices)
 *  5) Reads simulator objects' parameters from configuration file
 *  6) Simulation setup (Deseralization, Initailizing values, etc.)
 *  7) Run Simulation
 *  8) Simulation shutdown (Save results, serialize)
 *
 */
#include <string>

class Driver {
public:
   Driver() = default;   // default constructor

   Driver(const Driver &driver) = default;   // default copy constructor

   Driver &operator=(const Driver &driver) = default;   // default copy assignment

   Driver(Driver &&driver) = default;   // default move constructor

   Driver &operator=(Driver &&) = default;   // default move assignment

   ///  setupSimulation handles command line arguments and loads parameters
   ///  from parameter file. All initial loading before running simulator in Network
   ///  is done here.
   ///
   ///  @param  cmdLineArguments  command line arguments.
   ///  @return -1 if error, else 0 if success.
   int setupSimulation(std::string cmdLineArguments);
};
