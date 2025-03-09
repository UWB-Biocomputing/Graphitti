/**
 * @file Layout.cpp
 *
 * @ingroup Simulator/Layouts
 * 
 * @brief The Layout class defines the layout of neurons in neural networks
 */

#include "Layout.h"
#include "Factory.h"
#include "GraphManager.h"
#include "Global.h"
#include "OperationManager.h"
#include "ParameterManager.h"
#include "ParseParamError.h"
#include "RecordableBase.h"
#include "Simulator.h"
#include "Util.h"

/// Constructor
Layout::Layout() : numEndogenouslyActiveNeurons_(0)
{
   // Get a copy of the console logger to use in the case of errors
   log4cplus::Logger consoleLogger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("console"));

   // Create Vertices/Neurons class using type definition in configuration file
   string type;
   ParameterManager::getInstance().getStringByXpath("//VerticesParams/@class", type);
   vertices_ = Factory<AllVertices>::getInstance().createType(type);

   // If the factory returns an error (nullptr), exit
   if (vertices_ == nullptr) {
      LOG4CPLUS_INFO(consoleLogger, "INVALID CLASS: " + type);
      exit(EXIT_FAILURE);
   }

   // Register loadParameters function as a loadParameters operation in the Operation Manager
   function<void()> loadParametersFunc = std::bind(&Layout::loadParameters, this);
   OperationManager::getInstance().registerOperation(Operations::loadParameters,
                                                     loadParametersFunc);

   // Register printParameters function as a printParameters operation in the OperationManager
   function<void()> printParametersFunc = bind(&Layout::printParameters, this);
   OperationManager::getInstance().registerOperation(Operations::printParameters,
                                                     printParametersFunc);

   // Register registerGraphProperties method as registerGraphProperties operation
   // in the OperationManager
   function<void()> registerGraphPropertiesFunc = bind(&Layout::registerGraphProperties, this);
   OperationManager::getInstance().registerOperation((Operations::registerGraphProperties),
                                                     registerGraphPropertiesFunc);

   // Get a copy of the file logger to use log4cplus macros
   fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));
}

AllVertices &Layout::getVertices() const
{
   return *vertices_;
}

int Layout::getNumVertices() const
{
   return numVertices_;
}

/// Load member variables from configuration file. Registered to OperationManager as Operations::op::loadParameters
void Layout::loadParameters()
{
   numVertices_ = GraphManager<NeuralVertexProperties>::getInstance().numVertices();
}

void Layout::registerGraphProperties()
{
   GraphManager<VertexProperties> &gm = GraphManager<VertexProperties>::getInstance();
   gm.registerProperty("y", &VertexProperties::y);
   gm.registerProperty("x", &VertexProperties::x);
   gm.registerProperty("type", &VertexProperties::type);
}


/// Setup the internal structure of the class.
/// Allocate memories to store all layout state, no sequential dependency in this method
void Layout::setup()
{
   // Allocate memory
   xloc_ = VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, numVertices_);
   yloc_ = VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, numVertices_);
   dist2_ = CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, numVertices_, numVertices_);
   dist_ = CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, numVertices_, numVertices_);

   // more allocation of internal memory
   starterMap_.assign(numVertices_, false);
   vertexTypeMap_.assign(numVertices_, vertexType::VTYPE_UNDEF);

   // Loop over all vertices and set their x and y locations
   GraphManager<NeuralVertexProperties>::VertexIterator vi, vi_end;
   GraphManager<NeuralVertexProperties> &gm = GraphManager<NeuralVertexProperties>::getInstance();
   for (boost::tie(vi, vi_end) = gm.vertices(); vi != vi_end; ++vi) {
      assert(*vi < numVertices_);
      xloc_[*vi] = gm[*vi].x;
      yloc_[*vi] = gm[*vi].y;
   }

   // Now we calculate the distance and distance^2
   // between each pair of vertices
   for (int n = 0; n < numVertices_ - 1; n++) {
      for (int n2 = n + 1; n2 < numVertices_; n2++) {
         // distance^2 between two points in point-slope form
         dist2_(n, n2) = (xloc_[n] - xloc_[n2]) * (xloc_[n] - xloc_[n2])
                         + (yloc_[n] - yloc_[n2]) * (yloc_[n] - yloc_[n2]);

         // both points are equidistant from each other
         dist2_(n2, n) = dist2_(n, n2);
      }
   }

   // Finally take the square root to get the distances
   dist_ = sqrt(dist2_);

   // Register variable: vertex locations if need
   //Recorder &recorder = Simulator::getInstance().getModel().getRecorder();
   //string baseName = "Location";
   //string xLocation = "x_" + baseName;
   //string yLocation = "y_" + baseName;
   //recorder.registerVariable(xLocation, xloc_, Recorder::UpdatedType::CONSTANT);
   //recorder.registerVariable(yLocation, yloc_, Recorder::UpdatedType::CONSTANT);

   // test purpose
   // cout << "xloc_: " << &xloc_ << endl;
   // RecordableBase& location = xloc_;
   // cout << "location: " << &location << endl;
}


/// Prints out all parameters to logging file. Registered to OperationManager as Operation::printParameters
void Layout::printParameters() const
{
   GraphManager<NeuralVertexProperties>::VertexIterator vi, vi_end;
   GraphManager<NeuralVertexProperties> &gm = GraphManager<NeuralVertexProperties>::getInstance();
   stringstream output;
   output << "\nLAYOUT PARAMETERS" << endl;
   output << "\tEndogenously active neuron positions: ";

   for (boost::tie(vi, vi_end) = gm.vertices(); vi != vi_end; ++vi) {
      assert(*vi < numVertices_);
      if (gm[*vi].active) {
         output << *vi << " ";
      }
   }
   output << endl;

   output << "\tInhibitory neuron positions: ";

   for (boost::tie(vi, vi_end) = gm.vertices(); vi != vi_end; ++vi) {
      assert(*vi < numVertices_);
      if (gm[*vi].type == "INH") {
         output << *vi << " ";
      }
   }
   output << endl;

   LOG4CPLUS_DEBUG(fileLogger_, output.str());
}

/// Creates a vertex type map.
/// @param  numVertices number of the vertices to have in the type map.
void Layout::generateVertexTypeMap()
{
   DEBUG(cout << "\nInitializing vertex type map: VTYPE_UNDEF" << endl;);
   vertexTypeMap_.assign(numVertices_, vertexType::VTYPE_UNDEF);
}

/// Populates the starter map.
/// Selects num_endogenously_active_neurons excitory neurons and converts them into starter neurons.
/// @param  numVertices number of vertices to have in the map.
void Layout::initStarterMap()
{
   starterMap_.assign(numVertices_, false);
}
