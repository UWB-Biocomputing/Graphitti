/**
 * @file Model.cpp
 *
 * @ingroup Simulator/Core
 * 
 * @brief Implementation of Model for the graph-based networks.
 *
 * The network is composed of 3 superimposed 2-d arrays: vertices, edges, and
 * summation points.
 *
 * Edges in the edge map are located at the coordinates of the vertex
 * from which they receive output.  Each edge stores a pointer into a
 * summation point. 
 */

#include "Model.h"
#include "Connections.h"
#include "Factory.h"
#include "ParameterManager.h"
#include "Recorder.h"
#include "Simulator.h"

/// Constructor
Model::Model()
{
   // Get a copy of the console logger to use in the case of errors
   log4cplus::Logger consoleLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("console"));
   
   // Reference variable used to get class type from ParameterManager.
   string type;

   // Create Layout class using type definition from configuration file.
   ParameterManager::getInstance().getStringByXpath("//LayoutParams/@class", type);
   layout_ = Factory<Layout>::getInstance().createType(type);

   // If the factory returns an error (nullptr), exit
   if (layout_ == nullptr) {
      LOG4CPLUS_INFO(consoleLogger_, "INVALID CLASS: " + type);
      exit(EXIT_FAILURE);
   }

   // Create Connections class using type definition from configuration file.
   ParameterManager::getInstance().getStringByXpath("//ConnectionsParams/@class", type);
   connections_ = Factory<Connections>::getInstance().createType(type);

   // If the factory returns an error (nullptr), exit
   if (connections_ == nullptr) {
      LOG4CPLUS_INFO(consoleLogger_, "INVALID CLASS: " + type);
      exit(EXIT_FAILURE);
   }

   // Create Recorder class using type definition from configuration file.
   ParameterManager::getInstance().getStringByXpath("//RecorderParams/@class", type);
   recorder_ = Factory<Recorder>::getInstance().createType(type);

   // If the factory returns an error (nullptr), exit
   if (recorder_ == nullptr) {
      LOG4CPLUS_INFO(consoleLogger_, "INVALID CLASS: " + type);
      exit(EXIT_FAILURE);
   }

   // Get a copy of the file logger to use log4cplus macros
   fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));
}

/// Save simulation results to an output destination.
void Model::saveResults()
{
   if (recorder_ != nullptr) {
      recorder_->saveSimData(layout_->getVertices());
   }
}

/// Creates all the vertices and generates data for them.
// todo: this is going to go away
void Model::createAllVertices()
{
   LOG4CPLUS_INFO(fileLogger_, "Allocating Vertices...");

   layout_->generateVertexTypeMap(Simulator::getInstance().getTotalVertices());
   layout_->initStarterMap(Simulator::getInstance().getTotalVertices());

   // set their specific types
   layout_->getVertices().createAllVertices(*layout_);
}

/// Sets up the Simulation.
void Model::setupSim()
{
   LOG4CPLUS_INFO(fileLogger_, "Setting up Vertices...");
   layout_->getVertices().setupVertices();
   LOG4CPLUS_INFO(fileLogger_, "Setting up Edges...");
   connections_->getEdges().setupEdges();
#ifdef PERFORMANCE_METRICS
   // Start timer for initialization
   Simulator::getInstance().getShort_timer().start();
#endif
   LOG4CPLUS_INFO(fileLogger_, "Setting up Layout...");
   layout_->setup();
#ifdef PERFORMANCE_METRICS
   // Time to initialization (layout)
   t_host_initialization_layout += Simulator::getInstance().getShort_timer().lap() / 1000000.0;
#endif
   // Init radii and rates history matrices with default values
   if (recorder_ != nullptr) {
      recorder_->init();
      recorder_->initDefaultValues();
   }

   // Creates all the vertices and generates data for them.
   createAllVertices();

#ifdef PERFORMANCE_METRICS
   // Start timer for initialization
   Simulator::getInstance().getShort_timer().start();
#endif
   LOG4CPLUS_INFO(fileLogger_, "Setting up Connections...");
   connections_->setup();
#ifdef PERFORMANCE_METRICS
   // Time to initialization (connections)
   t_host_initialization_connections += Simulator::getInstance().getShort_timer().lap() / 1000000.0;
#endif

   // create an edge index map
   LOG4CPLUS_INFO(fileLogger_, "Creating EdgeIndexMap...");
   connections_->createEdgeIndexMap();
}

// Note: This method was previously used for debugging, but it is now dead code left behind.
/// Log this simulation step.
// void Model::logSimStep() const
// {
//    FixedLayout *fixedLayout = dynamic_cast<FixedLayout *>(layout_.get());
//    if (fixedLayout == nullptr) {
//       return;
//    }

//    fixedLayout->printLayout();
// }

/// Update the simulation history of every epoch.
void Model::updateHistory()
{
   // Compile history information in every epoch
   if (recorder_ == nullptr) {
      LOG4CPLUS_INFO(fileLogger_, "ERROR: Recorder class is null.");
   }
   if (recorder_ != nullptr) {
      recorder_->compileHistories(layout_->getVertices());
   }
}

/// Get the Connections class object.
/// @return Pointer to the Connections class object.
// ToDo: make smart ptr
Connections &Model::getConnections() const
{
   return *connections_;
}

/// Get the Layout class object.
/// @return Pointer to the Layout class object.
Layout &Model::getLayout() const
{
   return *layout_;
}

/// Get the Recorder class object.
/// @return Pointer to the Recorder class object.
// ToDo: make smart ptr
Recorder &Model::getRecorder() const
{
   return *recorder_;
}
