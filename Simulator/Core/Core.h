/**
 * @file Core.h
 * 
 * @ingroup Simulator/Core
 *
 * @brief Orchestrates most functionality in the simulation.
 * 
 *  The runSimulation performs the following steps:
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

class Core {
public:
   Core() = default;   // default constructor

   ///  runSimulation handles command line arguments and loads parameters
   ///  from parameter file. All initial config loading & running the simulator
   ///  is done here.
   ///
   ///  @param  cmdLineArguments  command line arguments.
   ///  @param  executableName Name of the simultaor's executable file
   ///  @return -1 if error, else 0 for success.
   int runSimulation(std::string executableName, std::string cmdLineArguments);

private:
   ///  Handles parsing of the command line
   ///
   ///  @param  cmdLineArguments command line argument
   ///  @param  executableName Name of the simultaor's executable file
   ///  @returns    true if successful, false otherwise.
   bool parseCommandLine(std::string executableName, std::string cmdLineArguments);
};
