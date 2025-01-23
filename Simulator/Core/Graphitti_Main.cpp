/**
 * @file Graphitti_Main.cpp
 * 
 * @ingroup Simulator/Core
 *
 * @brief Starting point of the Simulation - Main.
 * 
 *  The main functions calls the Core's runSimulation method which performs the following simulation steps:
 *  1) Instantiates Simulator object
 *  2) Parses command line to get configuration file and additional information if provided
 *  3) Loads global Simulator parameters from configuration file
 *  4) Instantiates all simulator objects (Layout, Connections, Synapases, Vertices)
 *  5) Reads simulator objects' parameters from configuration file
 *  6) Simulation setup (Deseralization, Initailizing values, etc.)
 *  7) Run Simulation
 *  8) Simulation shutdown (Save results, serialize)
 *
 * The Core is de-coupled from main to improve testability.
 */

#include "Core.h"
#include "log4cplus/configurator.h"
#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"
#include <fstream>
#include <iostream>
#include <string>
using namespace std;

///  Main function calls the Core's runSimulation method which
///  handles command line arguments and running the simulation.
///
///  @param  argc    argument count.
///  @param  argv    arguments.
///  @return -1 if error, else 0 if success.
int main(int argc, char *argv[])
{
   #ifdef TESTING_MODE
      cout << "Testing mode is ON \n";
   #endif 
   
   // Clear logging files at the start of each simulation
   fstream("Output/Debug/logging.txt", ios::out | ios::trunc);
   fstream("Output/Debug/vertices.txt", ios::out | ios::trunc);
   fstream("Output/Debug/edges.txt", ios::out | ios::trunc);

   // Initialize log4cplus and set properties based on configure file
   ::log4cplus::initialize();
   ::log4cplus::PropertyConfigurator::doConfigure("RuntimeFiles/log4cplus_configure.ini");

   // storing command line arguments as string
   // required to pass as an argument to setupSimulation
   string cmdLineArguments;
   string executableName = argv[0];
   for (int i = 1; i < argc; i++) {
      cmdLineArguments = cmdLineArguments + argv[i] + " ";
   }

   // Creating an instance of core class
   Core core;
   return core.runSimulation(executableName, cmdLineArguments);
};
