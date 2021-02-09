/**
 * @file Layout.cpp
 *
 * @ingroup Simulator/Layouts
 * 
 * @brief 
 */

#include "Layout.h"
#include "Simulator.h"
#include "ParseParamError.h"
#include "Util.h"
#include "ParameterManager.h"
#include "OperationManager.h"
#include "VerticiesFactory.h"

/// Constructor
Layout::Layout() :
      numEndogenouslyActiveNeurons_(0),
      gridLayout_(true) {
   xloc_ = NULL;
   yloc_ = NULL;
   dist2_ = NULL;
   dist_ = NULL;
   neuronTypeMap_ = NULL;
   starterMap_ = NULL;

   // Create Vertices/Neurons class using type definition in configuration file
   string type;
   ParameterManager::getInstance().getStringByXpath("//NeuronsParams/@class", type);
   neurons_ = VerticesFactory::getInstance()->createVertices(type);

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
   if (xloc_ != NULL) delete xloc_;
   if (yloc_ != NULL) delete yloc_;
   if (dist2_ != NULL) delete dist2_;
   if (dist_ != NULL) delete dist_;
   if (neuronTypeMap_ != NULL) delete[] neuronTypeMap_;  //todo: is delete[] changing once array becomes vector?
   if (starterMap_ != NULL) delete[] starterMap_; //todo: is delete[] changing once array becomes vector?

   xloc_ = NULL;
   yloc_ = NULL;
   dist2_ = NULL;
   dist_ = NULL;
   neuronTypeMap_ = NULL;
   starterMap_ = NULL;
}

shared_ptr<IAllVertices> Layout::getVertices() const {
   return neurons_;
}


/// Setup the internal structure of the class.
/// Allocate memories to store all layout state, no sequential dependency in this method
void Layout::setupLayout() {
   int numNeurons = Simulator::getInstance().getTotalVertices();

   // Allocate memory
   xloc_ = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, numNeurons);
   yloc_ = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, numNeurons);
   dist2_ = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, numNeurons, numNeurons);
   dist_ = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, numNeurons, numNeurons);

   // Initialize neuron locations memory, grab global info
   initNeuronsLocs();

   // computing distance between each pair of neurons given each neuron's xy location
   for (int n = 0; n < numNeurons - 1; n++) {
      for (int n2 = n + 1; n2 < numNeurons; n2++) {
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
   neuronTypeMap_ = new neuronType[numNeurons]; // todo: make array into vector
   starterMap_ = new bool[numNeurons]; // todo: make array into vector
}

/// Load member variables from configuration file. Registered to OperationManager as Operations::op::loadParameters
void Layout::loadParameters() {
   // Get the file paths for the Neuron lists from the configuration file
   string activeNListFilePath;
   string inhibitoryNListFilePath;
   if (!ParameterManager::getInstance().getStringByXpath("//LayoutFiles/activeNListFileName/text()",
                                                         activeNListFilePath)) {
      throw runtime_error("In Layout::loadParameters() Endogenously "
                          "active neuron list file path wasn't found and will not be initialized");
   }
   if (!ParameterManager::getInstance().getStringByXpath("//LayoutFiles/inhNListFileName/text()",
                                                         inhibitoryNListFilePath)) {
      throw runtime_error("In Layout::loadParameters() "
                          "Inhibitory neuron list file path wasn't found and will not be initialized");
   }

   // Initialize Neuron Lists based on the data read from the xml files
   if (!ParameterManager::getInstance().getIntVectorByXpath(activeNListFilePath, "A", endogenouslyActiveNeuronList_)) {
      throw runtime_error("In Layout::loadParameters() "
                          "Endogenously active neuron list file wasn't loaded correctly"
                          "\n\tfile path: " + activeNListFilePath);
   }
   numEndogenouslyActiveNeurons_ = endogenouslyActiveNeuronList_.size();
   if (!ParameterManager::getInstance().getIntVectorByXpath(inhibitoryNListFilePath, "I", inhibitoryNeuronLayout_)) {
      throw runtime_error("In Layout::loadParameters() "
                          "Inhibitory neuron list file wasn't loaded correctly."
                          "\n\tfile path: " + inhibitoryNListFilePath);
   }
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

/// Creates a neurons type map.
/// @param  numNeurons number of the neurons to have in the type map.
void Layout::generateNeuronTypeMap(int numNeurons) {
   DEBUG(cout << "\nInitializing neuron type map" << endl;);

   for (int i = 0; i < numNeurons; i++) {
      neuronTypeMap_[i] = EXC;
   }
}

/// Populates the starter map.
/// Selects num_endogenously_active_neurons excitory neurons and converts them into starter neurons.
/// @param  numNeurons number of neurons to have in the map.
void Layout::initStarterMap(const int numNeurons) {
   for (int i = 0; i < numNeurons; i++) {
      starterMap_[i] = false;
   }
}

///  Returns the type of synapse at the given coordinates
///
///  @param    srcNeuron  integer that points to a Neuron in the type map as a source.
///  @param    destNeuron integer that points to a Neuron in the type map as a destination.
///  @return type of the synapse.
synapseType Layout::synType(const int srcNeuron, const int destNeuron) {
   if (neuronTypeMap_[srcNeuron] == INH && neuronTypeMap_[destNeuron] == INH)
      return II;
   else if (neuronTypeMap_[srcNeuron] == INH && neuronTypeMap_[destNeuron] == EXC)
      return IE;
   else if (neuronTypeMap_[srcNeuron] == EXC && neuronTypeMap_[destNeuron] == INH)
      return EI;
   else if (neuronTypeMap_[srcNeuron] == EXC && neuronTypeMap_[destNeuron] == EXC)
      return EE;

   return STYPE_UNDEF;
}

/// Initialize the location maps (xloc and yloc).
void Layout::initNeuronsLocs() {
   int numNeurons = Simulator::getInstance().getTotalVertices();

   // Initialize neuron locations
   if (gridLayout_) {
      // grid layout
      for (int i = 0; i < numNeurons; i++) {
         (*xloc_)[i] = i % Simulator::getInstance().getHeight();
         (*yloc_)[i] = i / Simulator::getInstance().getHeight();
      }
   } else {
      // random layout
      for (int i = 0; i < numNeurons; i++) {
         (*xloc_)[i] = rng.inRange(0, Simulator::getInstance().getWidth());
         (*yloc_)[i] = rng.inRange(0, Simulator::getInstance().getHeight());
      }
   }
}



