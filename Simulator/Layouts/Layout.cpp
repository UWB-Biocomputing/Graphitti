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
#include "OperationManager.h"
#include "ParameterManager.h"
#include "ParseParamError.h"
#include "RecordableBase.h"
#include "Simulator.h"
#include "Util.h"

/// Constructor
Layout::Layout()
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

   function<void()> registerHistoryVariablesFunc = bind(&Layout::registerHistoryVariables, this);
   OperationManager::getInstance().registerOperation(Operations::registerHistoryVariables,
                                                     registerHistoryVariablesFunc);

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
   GraphManager<NeuralVertexProperties> &gm = GraphManager<NeuralVertexProperties>::getInstance();
   gm.registerProperty("y", &VertexProperties::y);
   gm.registerProperty("x", &VertexProperties::x);
   gm.registerProperty("type", &VertexProperties::type);
}

void Layout::registerHistoryVariables()
{
   // Register vertex type map
   Recorder &recorder = Simulator::getInstance().getModel().getRecorder();
   recorder.registerVariable("vertexTypeMap", vertexTypeMap_, Recorder::UpdatedType::CONSTANT);
}


/// Setup the internal structure of the class.
/// Allocate memories to store all layout state, no sequential dependency in this method
void Layout::setup()
{
   // Allocation of internal memory
   vertexTypeMap_.assign(numVertices_, vertexType::VTYPE_UNDEF);
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

void Layout::initStarterMap()
{
}
