/**
 * @file LayoutNeuro.cpp
 *
 * @ingroup Simulator/Layouts
 * 
 * @brief The Layout class defines the layout of vertices in neural networks
 */

#include "LayoutNeuro.h"
#include "ConnGrowth.h"
#include "GraphManager.h"
#include "ParameterManager.h"
#include "ParseParamError.h"
#include "Util.h"

// TODO: I don't think that either of the constructor or destructor is needed here
LayoutNeuro::LayoutNeuro() : Layout()
{
}

// Register vertex properties with the GraphManager
void LayoutNeuro::registerGraphProperties()
{
   // The base class registers properties that are common to all vertices
   Layout::registerGraphProperties();

   // We must register the graph properties before loading it.
   // We are passing a pointer to a data member of the VertexProperty
   // so Boost Graph Library can use it for loading the graphML file.
   // Look at: https://www.studytonight.com/cpp/pointer-to-members.php
   GraphManager<NeuralVertexProperties> &gm = GraphManager<NeuralVertexProperties>::getInstance();
   gm.registerProperty("y", &NeuralVertexProperties::y);
   gm.registerProperty("x", &NeuralVertexProperties::x);
   gm.registerProperty("type", &NeuralVertexProperties::type);
   gm.registerProperty("active", &NeuralVertexProperties::active);

   // gm.registerProperty("source", &NeuralEdgeProperties::source);
   // gm.registerProperty("target", &NeuralEdgeProperties::target);
   // gm.registerProperty("weight", &NeuralEdgeProperties::weight);

}

///  Prints out all parameters to logging file.
///  Registered to OperationManager as Operation::printParameters
void LayoutNeuro::printParameters() const
{
   Layout::printParameters();

   LOG4CPLUS_DEBUG(fileLogger_, "\n\tLayout type: LayoutNeuro" << endl << endl);
}

///  Creates a randomly ordered distribution with the specified numbers of vertex types.
///
///  @param  numVertices number of the vertices to have in the type map.
void LayoutNeuro::generateVertexTypeMap()
{
   LOG4CPLUS_DEBUG(fileLogger_, "\nInitializing vertex type map" << endl);

   int numInhibitoryNeurons;
   int numExcititoryNeurons;

   // Set Neuron Type from GraphML File
   GraphManager<NeuralVertexProperties>::VertexIterator vi, vi_end;
   GraphManager<NeuralVertexProperties> &gm = GraphManager<NeuralVertexProperties>::getInstance();

   for (boost::tie(vi, vi_end) = gm.vertices(); vi != vi_end; ++vi) {
      assert(*vi < numVertices_);
      if (gm[*vi].type == "INH") {
         vertexTypeMap_[*vi] = vertexType::INH;
         numInhibitoryNeurons++;
      }
      // Default Type is Excitatory
      else {
         vertexTypeMap_[*vi] = vertexType::EXC;
      }
   }

   numExcititoryNeurons = numVertices_ - numInhibitoryNeurons;

   LOG4CPLUS_DEBUG(fileLogger_, "\nVERTEX TYPE MAP"
                                   << endl
                                   << "\tTotal vertices: " << numVertices_ << endl
                                   << "\tInhibitory Neurons: " << numInhibitoryNeurons << endl
                                   << "\tExcitatory Neurons: " << numExcititoryNeurons << endl);
   LOG4CPLUS_INFO(fileLogger_, "Finished initializing vertex type map");
}

///  Populates the starter map.
///  Selects \e numStarter excitory neurons and converts them into starter neurons.
///  @param  numVertices number of vertices to have in the map.
void LayoutNeuro::initStarterMap()
{
   Layout::initStarterMap();

   // Set Neuron Activity from GraphML File
   GraphManager<NeuralVertexProperties>::VertexIterator vi, vi_end;
   GraphManager<NeuralVertexProperties> &gm = GraphManager<NeuralVertexProperties>::getInstance();

   for (boost::tie(vi, vi_end) = gm.vertices(); vi != vi_end; ++vi) {
      assert(*vi < numVertices_);
      if (gm[*vi].active) {
         starterMap_[*vi] = true;
         numEndogenouslyActiveNeurons_++;
      }
   }
}

///  Returns the type of synapse at the given coordinates
///
///  @param    srcVertex  integer that points to a Neuron in the type map as a source.
///  @param    destVertex integer that points to a Neuron in the type map as a destination.
///  @return type of the synapse.
edgeType LayoutNeuro::edgType(int srcVertex, int destVertex)
{
   if (vertexTypeMap_[srcVertex] == vertexType::INH
       && vertexTypeMap_[destVertex] == vertexType::INH)
      return edgeType::II;
   else if (vertexTypeMap_[srcVertex] == vertexType::INH
            && vertexTypeMap_[destVertex] == vertexType::EXC)
      return edgeType::IE;
   else if (vertexTypeMap_[srcVertex] == vertexType::EXC
            && vertexTypeMap_[destVertex] == vertexType::INH)
      return edgeType::EI;
   else if (vertexTypeMap_[srcVertex] == vertexType::EXC
            && vertexTypeMap_[destVertex] == vertexType::EXC)
      return edgeType::EE;

   return edgeType::ETYPE_UNDEF;
}
