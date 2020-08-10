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

   // Register loadParameters function with Operation Manager
   auto function = std::bind(&Layout::loadParameters, this);
   OperationManager::getInstance().registerOperation(Operations::op::loadParameters, function);
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

shared_ptr<IAllNeurons> Layout::getNeurons() const {
   return neurons_;
}


/// Setup the internal structure of the class.
/// Allocate memories to store all layout state, no sequential dependency in this method
void Layout::setupLayout() {
   int num_neurons = Simulator::getInstance().getTotalNeurons();

   // Allocate memory
   xloc_ = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons);
   yloc_ = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons);
   dist2_ = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons);
   dist_ = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons);

   // Initialize neuron locations memory, grab global info
   initNeuronsLocs();

   // computing distance between each pair of neurons given each neuron's xy location
   for (int n = 0; n < num_neurons - 1; n++) {
      for (int n2 = n + 1; n2 < num_neurons; n2++) {
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
   neuronTypeMap_ = new neuronType[num_neurons]; // todo: make array into vector
   starterMap_ = new bool[num_neurons]; // todo: make array into vector
}

/// Load member variables from configuration file. Registered to OperationManager as Operations::op::loadParameters
void Layout::loadParameters() {
   // Get the file paths for the Neuron lists from the configuration file
   string activeNListFilePath;
   string inhibitoryNListFilePath;
   if (!ParameterManager::getInstance().getStringByXpath("//LayoutFiles/activeNListFileName/text()",
                                                         activeNListFilePath)) {
      cerr << "In Layout::loadParameters() "
           << "Endogenously active neuron list file path wasn't found and will not be initialized"
           << endl;
      return;
   }
   if (!ParameterManager::getInstance().getStringByXpath("//LayoutFiles/inhNListFileName/text()",
                                                         inhibitoryNListFilePath)) {
      cerr << "In Layout::loadParameters() "
           << "Inhibitory neuron list file path wasn't found and will not be initialized"
           << endl;
      return;
   }

   // Initialize Neuron Lists based on the data read from the xml files
   if (!ParameterManager::getInstance().getIntVectorByXpath(activeNListFilePath, "A", endogenouslyActiveNeuronList_)) {
      cerr << "In Layout::loadParameters() "
           << "Endogenously active neuron list file wasn't loaded correctly"
           << "\n\tfile path: " << activeNListFilePath << endl;
      return;
   }
   numEndogenouslyActiveNeurons_ = endogenouslyActiveNeuronList_.size();
   if (!ParameterManager::getInstance().getIntVectorByXpath(inhibitoryNListFilePath, "I", inhibitoryNeuronLayout_)) {
      cerr << "In Layout::loadParameters() "
           << "Inhibitory neuron list file wasn't loaded correctly."
            << "\n\tfile path: " << inhibitoryNListFilePath << endl;
      return;
   }
}


/// Prints out all parameters of the layout to console.
void Layout::printParameters() const {
}

/// Creates a neurons type map.
/// @param  num_neurons number of the neurons to have in the type map.
void Layout::generateNeuronTypeMap(int num_neurons) {
   DEBUG(cout << "\nInitializing neuron type map" << endl;);

   for (int i = 0; i < num_neurons; i++) {
      neuronTypeMap_[i] = EXC;
   }
}

/// Populates the starter map.
/// Selects num_endogenously_active_neurons excitory neurons and converts them into starter neurons.
/// @param  num_neurons number of neurons to have in the map.
void Layout::initStarterMap(const int num_neurons) {
   for (int i = 0; i < num_neurons; i++) {
      starterMap_[i] = false;
   }
}

/*
 *  Returns the type of synapse at the given coordinates
 *
 *  @param    src_neuron  integer that points to a Neuron in the type map as a source.
 *  @param    dest_neuron integer that points to a Neuron in the type map as a destination.
 *  @return type of the synapse.
 */
synapseType Layout::synType(const int src_neuron, const int dest_neuron) {
   if (neuronTypeMap_[src_neuron] == INH && neuronTypeMap_[dest_neuron] == INH)
      return II;
   else if (neuronTypeMap_[src_neuron] == INH && neuronTypeMap_[dest_neuron] == EXC)
      return IE;
   else if (neuronTypeMap_[src_neuron] == EXC && neuronTypeMap_[dest_neuron] == INH)
      return EI;
   else if (neuronTypeMap_[src_neuron] == EXC && neuronTypeMap_[dest_neuron] == EXC)
      return EE;

   return STYPE_UNDEF;
}

/// Initialize the location maps (xloc and yloc).
void Layout::initNeuronsLocs() {
   int num_neurons = Simulator::getInstance().getTotalNeurons();

   // Initialize neuron locations
   if (gridLayout_) {
      // grid layout
      for (int i = 0; i < num_neurons; i++) {
         (*xloc_)[i] = i % Simulator::getInstance().getHeight();
         (*yloc_)[i] = i / Simulator::getInstance().getHeight();
      }
   } else {
      // random layout
      for (int i = 0; i < num_neurons; i++) {
         (*xloc_)[i] = rng.inRange(0, Simulator::getInstance().getWidth());
         (*yloc_)[i] = rng.inRange(0, Simulator::getInstance().getHeight());
      }
   }
}



