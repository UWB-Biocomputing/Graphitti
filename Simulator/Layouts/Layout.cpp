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
}

/// Setup the internal structure of the class.
/// Allocate memories to store all layout state, no sequential dependency in this method
void Layout::setup()
{
   dist2_ = CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, numVertices_, numVertices_);
   dist_ = CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, numVertices_, numVertices_);
}


/// Creates a vertex type map.
/// @param  numVertices number of the vertices to have in the type map.
void Layout::generateVertexTypeMap()
{
   DEBUG(cout << "\nInitializing vertex type map: VTYPE_UNDEF" << endl;);
   getVertices().vertexTypeMap_.assign(numVertices_, vertexType::VTYPE_UNDEF);
}

void Layout::initStarterMap()
{
}
