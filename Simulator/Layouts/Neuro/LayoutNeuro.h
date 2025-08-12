/**
 * @file LayoutNeuro.h
 * 
 * @ingroup Simulator/Layouts
 *
 * @brief The Layout class defines the layout of vertices in neural networks
 *
 * The LayoutNeuro class maintains vertices locations (x, y coordinates), 
 * distance of every couple vertices,
 * vertices type map (distribution of excitatory and inhibitory neurons), and starter vertices map
 * (distribution of endogenously active neurons).  
 *
 * The LayoutNeuro class reads all layout information from parameter description file.
 */

#pragma once

#include "Layout.h"
// cereal
#include <cereal/types/polymorphic.hpp>

using namespace std;

class LayoutNeuro : public Layout {
public:
   LayoutNeuro();

   virtual ~LayoutNeuro() = default;

   static Layout *Create()
   {
      return new LayoutNeuro();
   }

   /// Register vertex properties with the GraphManager
   virtual void registerGraphProperties() override;

   virtual void registerHistoryVariables() override;

   /// Setup the internal structure of the class.
   /// Allocate memories to store all layout state.
   virtual void setup() override;

   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() const override;

   ///  Creates a vertex type map.
   virtual void generateVertexTypeMap() override;

   ///  Populates the starter map.
   ///  Selects num_endogenously_active_neurons excitory neurons
   ///  and converts them into starter vertices.
   virtual void initStarterMap() override;

   /// Returns the type of synapse at the given coordinates
   /// @param    srcVertex  integer that points to a Neuron in the type map as a source.
   /// @param    destVertex integer that points to a Neuron in the type map as a destination.
   /// @return type of the synapse.
   virtual edgeType edgType(int srcVertex, int destVertex) override;

   /// Prints the layout, used for debugging.
   void printLayout();

   VectorMatrix xloc_;   ///< Store neuron i's x location.

   VectorMatrix yloc_;   ///< Store neuron i's y location.

   vector<bool> starterMap_;   ///< The starter existence map (T/F).

   BGSIZE numEndogenouslyActiveNeurons_;   ///< Number of endogenously active neurons.

   ///  Cereal serialization method
   template <class Archive> void serialize(Archive &archive);
};

CEREAL_REGISTER_TYPE(LayoutNeuro);

///  Cereal serialization method
template <class Archive> void LayoutNeuro::serialize(Archive &archive)
{
   archive(cereal::virtual_base_class<Layout>(this), cereal::make_nvp("xloc", xloc_),
           cereal::make_nvp("yloc", yloc_), cereal::make_nvp("starterMap", starterMap_),
           cereal::make_nvp("numEndogenouslyActiveNeurons", numEndogenouslyActiveNeurons_));
}
