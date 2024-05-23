/**
 * @file LayoutNeuro.cpp
 *
 * @ingroup Simulator/Layouts
 * 
 * @brief The Layout class defines the layout of vertices in neural networks
 */

#include "LayoutNeuro.h"
#include "ConnGrowth.h"
#include "ParameterManager.h"
#include "GraphManager.h"
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
    // TODO: Currently not fully implemented because Neuro model is still integrating graphML
    Layout::registerGraphProperties();

    // We must register the graph properties before loading it.
    // We are passing a pointer to a data member of the VertexProperty
    // so Boost Graph Library can use it for loading the graphML file.
    // Look at: https://www.studytonight.com/cpp/pointer-to-members.php
    GraphManager& gm = GraphManager::getInstance();
    gm.registerProperty("active", &VertexProperty::active);
}

void LayoutNeuro::setup()
{
   // Base class allocates memory for: xLoc_, yLoc, dist2_, and dist_
   // so we call its method first
   Layout::setup();

   // Initialize neuron locations memory, grab global info
   initVerticesLocs();

   // computing distance between each pair of vertices given each vertex's xy location
   for (int n = 0; n < numVertices_ - 1; n++) {
      for (int n2 = n + 1; n2 < numVertices_; n2++) {
         // distance^2 between two points in point-slope form
         dist2_(n, n2) = (xloc_[n] - xloc_[n2]) * (xloc_[n] - xloc_[n2])
                         + (yloc_[n] - yloc_[n2]) * (yloc_[n] - yloc_[n2]);

         // both points are equidistant from each other
         dist2_(n2, n) = dist2_(n, n2);
      }
   }

   // take the square root to get actual distance (Pythagoras was right!)
   dist_ = sqrt(dist2_);
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
void LayoutNeuro::generateVertexTypeMap(int numVertices)
{
   LOG4CPLUS_DEBUG(fileLogger_, "\nInitializing vertex type map" << endl);

   int numInhibitoryNeurons;
   int numExcititoryNeurons;

   // Set Neuron Type from GraphML File
   GraphManager::VertexIterator vi, vi_end;
   GraphManager& gm = GraphManager::getInstance();

   for (boost::tie(vi, vi_end) = gm.vertices(); vi != vi_end; ++vi) {
       assert(*vi < numVertices_);
       if (gm[*vi].type == "INH") {
           vertexTypeMap_[*vi] = INH;
           numInhibitoryNeurons++;
       }
       // Default Type is Excitatory
       else {
           vertexTypeMap_[*vi] = EXC;
       }
   }

   numExcititoryNeurons = numVertices - numInhibitoryNeurons;

   LOG4CPLUS_DEBUG(fileLogger_, "\nVERTEX TYPE MAP"
                                   << endl
                                   << "\tTotal vertices: " << numVertices << endl
                                   << "\tInhibitory Neurons: " << numInhibitoryNeurons << endl
                                   << "\tExcitatory Neurons: " << numExcititoryNeurons << endl);
   LOG4CPLUS_INFO(fileLogger_, "Finished initializing vertex type map");
}

///  Populates the starter map.
///  Selects \e numStarter excitory neurons and converts them into starter neurons.
///  @param  numVertices number of vertices to have in the map.
void LayoutNeuro::initStarterMap(int numVertices)
{   
   Layout::initStarterMap(numVertices);
   
   // Set Neuron Activity from GraphML File
   GraphManager::VertexIterator vi, vi_end;
   GraphManager& gm = GraphManager::getInstance();

   for (boost::tie(vi, vi_end) = gm.vertices(); vi != vi_end; ++vi) {
       assert(*vi < numVertices_);
       if (gm[*vi].active) {
           starterMap_[*vi] = true;
           numEndogenouslyActiveNeurons_++;
       }
   }
}

/// Load member variables from configuration file. Registered to OperationManager as Operations::op::loadParameters
void LayoutNeuro::loadParameters()
{
   numVertices_ = GraphManager::getInstance().numVertices();
}

///  Returns the type of synapse at the given coordinates
///
///  @param    srcVertex  integer that points to a Neuron in the type map as a source.
///  @param    destVertex integer that points to a Neuron in the type map as a destination.
///  @return type of the synapse.
edgeType LayoutNeuro::edgType(int srcVertex, int destVertex)
{
   if (vertexTypeMap_[srcVertex] == INH && vertexTypeMap_[destVertex] == INH)
      return II;
   else if (vertexTypeMap_[srcVertex] == INH && vertexTypeMap_[destVertex] == EXC)
      return IE;
   else if (vertexTypeMap_[srcVertex] == EXC && vertexTypeMap_[destVertex] == INH)
      return EI;
   else if (vertexTypeMap_[srcVertex] == EXC && vertexTypeMap_[destVertex] == EXC)
      return EE;

   return ETYPE_UNDEF;
}

/// Initialize the location maps (xloc and yloc).
void LayoutNeuro::initVerticesLocs()
{
    // Loop over all vertices and set their x and y locations
   GraphManager::VertexIterator vi, vi_end;

   GraphManager &gm = GraphManager::getInstance();
   
   for (boost::tie(vi, vi_end) = gm.vertices(); vi != vi_end; ++vi) {
      assert(*vi < numVertices_);
      xloc_[*vi] = gm[*vi].x;
      yloc_[*vi] = gm[*vi].y;
   }
   
}

// Note: This code was previously used for debugging, but it is now dead code left behind
// and it is never executed.
/*void LayoutNeuro::printLayout()
{
   ConnGrowth &pConnGrowth
      = dynamic_cast<ConnGrowth &>(Simulator::getInstance().getModel().getConnections());

   cout << "format:\ntype,radius,firing rate" << endl;

   for (int y = 0; y < height_; y++) {
      stringstream ss;
      ss << fixed;
      ss.precision(1);

      for (int x = 0; x < width_; x++) {
         switch (vertexTypeMap_[x + y * width_]) {
            case EXC:
               if (starterMap_[x + y * width_])
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

         ss << " " << pConnGrowth.radii_[x + y * width_];

         if (x + 1 < width_) {
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
}*/
