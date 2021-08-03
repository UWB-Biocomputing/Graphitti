/**
 * @file Layout.cpp
 *
 * @ingroup Simulator/Layouts
 * 
 * @brief The Layout class defines the layout of neurons in neural networks
 */

#include "Layout.h"
#include "Simulator.h"
#include "ParseParamError.h"
#include "Util.h"
#include "ParameterManager.h"
#include "OperationManager.h"
#include "VerticesFactory.h"

/// Constructor
Layout::Layout() :
      numEndogenouslyActiveNeurons_(0),
      gridLayout_(true) {
   xloc_ = nullptr;
   yloc_ = nullptr;
   dist2_ = nullptr;
   dist_ = nullptr;
   vertexTypeMap_ = nullptr;
   starterMap_ = nullptr;

   // Create Vertices/Neurons class using type definition in configuration file
   string type;
   ParameterManager::getInstance().getStringByXpath("//VerticesParams/@class", type);
   vertices_ = VerticesFactory::getInstance()->createVertices(type);

   // Register loadParameters function as a loadParameters operation in the Operation Manager
   function<void()> loadParametersFunc = std::bind(&Layout::loadParameters, this);
   OperationManager::getInstance().registerOperation(Operations::op::loadParameters, loadParametersFunc);

   // Register printParameters function as a printParameters operation in the OperationManager
   function<void()> printParametersFunc = bind(&Layout::printParameters, this);
   OperationManager::getInstance().registerOperation(Operations::printParameters, printParametersFunc);

   // Get a copy of the file logger to use log4cplus macros
   fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));
}

/// Destructor
Layout::~Layout() {
   if (xloc_ != nullptr) delete xloc_;
   if (yloc_ != nullptr) delete yloc_;
   if (dist2_ != nullptr) delete dist2_;
   if (dist_ != nullptr) delete dist_;
   if (vertexTypeMap_ != nullptr) delete[] vertexTypeMap_;  //todo: is delete[] changing once array becomes vector?
   if (starterMap_ != nullptr) delete[] starterMap_; //todo: is delete[] changing once array becomes vector?

   xloc_ = nullptr;
   yloc_ = nullptr;
   dist2_ = nullptr;
   dist_ = nullptr;
   vertexTypeMap_ = nullptr;
   starterMap_ = nullptr;
}

shared_ptr<AllVertices> Layout::getVertices() const {
   return vertices_;
}


/// Setup the internal structure of the class.
/// Allocate memories to store all layout state, no sequential dependency in this method
void Layout::setupLayout() {
   int numVertices = Simulator::getInstance().getTotalVertices();

   // Allocate memory
   xloc_ = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, numVertices);
   yloc_ = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, numVertices);
   dist2_ = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, numVertices, numVertices);
   dist_ = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, numVertices, numVertices);

   // Initialize neuron locations memory, grab global info
   initVerticesLocs();

   // computing distance between each pair of vertices given each vertex's xy location
   for (int n = 0; n < numVertices - 1; n++) {
      for (int n2 = n + 1; n2 < numVertices; n2++) {
         // distance^2 between two points in point-slope form
         (*dist2_)(n, n2) = ((*xloc_)[n] - (*xloc_)[n2]) * ((*xloc_)[n] - (*xloc_)[n2]) +
                            ((*yloc_)[n] - (*yloc_)[n2]) * ((*yloc_)[n] - (*yloc_)[n2]);

         // both points are equidistant from each other
         (*dist2_)(n2, n) = (*dist2_)(n, n2);
      }
   }

   // take the square root to get actual distance (Pythagoras was right!)
   (*dist_) = sqrt((*dist2_));

   // more allocation of internal memory
   vertexTypeMap_ = new vertexType[numVertices]; // todo: make array into vector
   starterMap_ = new bool[numVertices]; // todo: make array into vector
}


/// Prints out all parameters to logging file. Registered to OperationManager as Operation::printParameters
void Layout::printParameters() const {
   stringstream output;
   output << "\nLAYOUT PARAMETERS" << endl;
   output << "\tEndogenously active neuron positions: ";
   for (BGSIZE i = 0; i < numEndogenouslyActiveNeurons_; i++) {
       output << endogenouslyActiveNeuronList_[i] << " ";
   }
   output << endl;

   output << "\tInhibitory neuron positions: ";
   for (BGSIZE i = 0; i < inhibitoryNeuronLayout_.size(); i++) {
      output << inhibitoryNeuronLayout_[i] << " ";
   }
   output << endl;

   LOG4CPLUS_DEBUG(fileLogger_, output.str());
}

/// Creates a vertex type map.
/// @param  numVertices number of the vertices to have in the type map.
void Layout::generateVertexTypeMap(int numVertices) {
   DEBUG(cout << "\nInitializing vertex type map: VTYPE_UNDEF" << endl;);

   for (int i = 0; i < numVertices; i++) {
      vertexTypeMap_[i] = VTYPE_UNDEF;
   }
}

/// Populates the starter map.
/// Selects num_endogenously_active_neurons excitory neurons and converts them into starter neurons.
/// @param  numVertices number of vertices to have in the map.
void Layout::initStarterMap(const int numVertices) {
   for (int i = 0; i < numVertices; i++) {
      starterMap_[i] = false;
   }
}

/// Initialize the location maps (xloc and yloc).
void Layout::initVerticesLocs() {
   int numVertices = Simulator::getInstance().getTotalVertices();

   // Initialize vertex locations
   if (gridLayout_) {
      // grid layout
      for (int i = 0; i < numVertices; i++) {
         (*xloc_)[i] = i % Simulator::getInstance().getHeight();
         (*yloc_)[i] = i / Simulator::getInstance().getHeight();
      }
   } else {
      // random layout
      for (int i = 0; i < numVertices; i++) {
         (*xloc_)[i] = initRNG.inRange(0, Simulator::getInstance().getWidth());
         (*yloc_)[i] = initRNG.inRange(0, Simulator::getInstance().getHeight());
      }
   }
}



