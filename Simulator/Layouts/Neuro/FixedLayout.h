/**
 * @file FixedLayout.h
 * 
 * @ingroup Simulator/Layouts
 *
 * @brief The Layout class defines the layout of vertices in neural networks
 *
 * The FixedLayout class maintains vertices locations (x, y coordinates), 
 * distance of every couple vertices,
 * vertices type map (distribution of excitatory and inhibitory neurons), and starter vertices map
 * (distribution of endogenously active neurons).  
 *
 * The FixedLayout class reads all layout information from parameter description file.
 */

#pragma once

#include "Layout.h"
// cereal
#include <cereal/types/polymorphic.hpp>

using namespace std;

class FixedLayout : public Layout {
public:
   FixedLayout();

   virtual ~FixedLayout() = default;

   static Layout *Create()
   {
      return new FixedLayout();
   }

   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const override;

   /// Setup the internal structure of the class.
   /// Allocate memories to store all layout state.
   virtual void setup() override;

   ///  Creates a vertex type map.
   ///
   ///  @param  numVertices number of the vertices to have in the type map.
   virtual void generateVertexTypeMap(int numVertices) override;

   ///  Populates the starter map.
   ///  Selects num_endogenously_active_neurons excitory neurons
   ///  and converts them into starter vertices.
   ///
   ///  @param  numVertices number of vertices to have in the map.
   virtual void initStarterMap(int numVertices) override;

   /// Load member variables from configuration file. Registered to OperationManager as Operation::loadParameters
   virtual void loadParameters() override;

   /// Returns the type of synapse at the given coordinates
   /// @param    srcVertex  integer that points to a Neuron in the type map as a source.
   /// @param    destVertex integer that points to a Neuron in the type map as a destination.
   /// @return type of the synapse.
   virtual edgeType edgType(int srcVertex, int destVertex) override;

   /// Prints the layout, used for debugging.
   void printLayout();

   ///  Cereal serialization method
   template <class Archive> void serialize(Archive &archive);

private:
   /// initialize the location maps (xloc and yloc).
   void initVerticesLocs();

   bool gridLayout_;   /// True if grid layout.

   int width_;   /// Width of the layout (assumes square)

   int height_;   /// Height of the layout
};

CEREAL_REGISTER_TYPE(FixedLayout);

///  Cereal serialization method
template <class Archive> void FixedLayout::serialize(Archive &archive)
{
   archive(cereal::base_class<Layout>(this), cereal::make_nvp("gridLayout_", gridLayout_),
           cereal::make_nvp("width_", width_), cereal::make_nvp("height_", height_));
}
