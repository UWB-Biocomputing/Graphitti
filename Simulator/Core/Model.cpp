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
#include "IRecorder.h"
#include "Connections.h"
#include "ConnGrowth.h"
#include "ParameterManager.h"
#include "LayoutFactory.h"
#include "ConnectionsFactory.h"
#include "RecorderFactory.h"

/// Constructor
Model::Model() {
   // Reference variable used to get class type from ParameterManager.
   string type;

   // Create Layout class using type definition from configuration file.
   ParameterManager::getInstance().getStringByXpath("//LayoutParams/@class", type);
   layout_ = LayoutFactory::getInstance()->createLayout(type);

   // Create Connections class using type definition from configuration file.
   ParameterManager::getInstance().getStringByXpath("//ConnectionsParams/@class", type);
   connections_ = ConnectionsFactory::getInstance()->createConnections(type);

   // Create Recorder class using type definition from configuration file.
   ParameterManager::getInstance().getStringByXpath("//RecorderParams/@class", type);
   recorder_ = RecorderFactory::getInstance()->createRecorder(type);

   // Get a copy of the file logger to use log4cplus macros
   fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));
}

/// Destructor
Model::~Model() {

}

/// Save simulation results to an output destination.
void Model::saveResults() {
   if (recorder_ != nullptr) {
      recorder_->saveSimData(*layout_->getVertices());
   }
}

/// Creates all the vertices and generates data for them.
// todo: this is going to go away
void Model::createAllVertices() {
   LOG4CPLUS_INFO(fileLogger_, "Allocating Vertices..." );

   layout_->generateVertexTypeMap(Simulator::getInstance().getTotalVertices());
   layout_->initStarterMap(Simulator::getInstance().getTotalVertices());

   // set their specific types
   layout_->getVertices()->createAllVertices(layout_.get());
}

/// Sets up the Simulation.
void Model::setupSim() {
   LOG4CPLUS_INFO(fileLogger_, "Setting up Vertices...");
   layout_->getVertices()->setupVertices();
   LOG4CPLUS_INFO(fileLogger_, "Setting up Edges...");
   connections_->getEdges()->setupEdges();
#ifdef PERFORMANCE_METRICS
   // Start timer for initialization
   Simulator::getInstance.short_timer.start();
#endif
   LOG4CPLUS_INFO(fileLogger_, "Setting up Layout...");
   layout_->setupLayout();
#ifdef PERFORMANCE_METRICS
   // Time to initialization (layout)
   t_host_initialization_layout += Simulator::getInstance().short_timer.lap() / 1000000.0;
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
   Simulator::getInstance().short_timer.start();
#endif
   LOG4CPLUS_INFO(fileLogger_, "Setting up Connections...");
   connections_->setupConnections(layout_.get(), layout_->getVertices().get(), connections_->getEdges().get());
#ifdef PERFORMANCE_METRICS
   // Time to initialization (connections)
   t_host_initialization_connections += Simulator::getInstance().short_timer.lap() / 1000000.0;
#endif

   // create an edge index map
   LOG4CPLUS_INFO(fileLogger_, "Creating EdgeIndexMap...");
   connections_->createEdgeIndexMap();
}

/// Log this simulation step.
void Model::logSimStep() const {
   ConnGrowth *pConnGrowth = dynamic_cast<ConnGrowth *>(connections_.get());
   if (pConnGrowth == nullptr)
      return;

   cout << "format:\ntype,radius,firing rate" << endl;

   for (int y = 0; y < Simulator::getInstance().getHeight(); y++) {
      stringstream ss;
      ss << fixed;
      ss.precision(1);

      for (int x = 0; x < Simulator::getInstance().getWidth(); x++) {
         switch (layout_->vertexTypeMap_[x + y * Simulator::getInstance().getWidth()]) {
            case EXC:
               if (layout_->starterMap_[x + y * Simulator::getInstance().getWidth()])
                  ss << "s";
               else
                  ss << "e";
               break;
            case INH:
               ss << "i";
               break;
            case VTYPE_UNDEF:
               assert(false);
               break;
         }

         ss << " " << (*pConnGrowth->radii_)[x + y * Simulator::getInstance().getWidth()];

         if (x + 1 < Simulator::getInstance().getWidth()) {
            ss.width(2);
            ss << "|";
            ss.width(2);
         }
      }

      ss << endl;

      for (int i = ss.str().length() - 1; i >= 0; i--) {
         ss << "_";
      }

      ss << endl;
      cout << ss.str();
   }
}

/// Update the simulation history of every epoch.
void Model::updateHistory() {
   // Compile history information in every epoch
      if(recorder_ == nullptr)
   {
      LOG4CPLUS_INFO(fileLogger_, "ERROR: Recorder class is null.");
   }
   if (recorder_ != nullptr) {
      recorder_->compileHistories(*layout_->getVertices());
   }
}

/// Get the Connections class object.
/// @return Pointer to the Connections class object.  
// ToDo: make smart ptr
shared_ptr<Connections> Model::getConnections() const { return connections_; }

/// Get the Layout class object.
/// @return Pointer to the Layout class object. 
// ToDo: make smart ptr
shared_ptr<Layout> Model::getLayout() const { return layout_; }

/// Get the IRecorder class object.
/// @return Pointer to the IRecorder class object. 
// ToDo: make smart ptr
shared_ptr<IRecorder> Model::getRecorder() const { return recorder_; }
