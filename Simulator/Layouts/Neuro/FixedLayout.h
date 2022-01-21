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

class FixedLayout : public Layout {
	public:
		FixedLayout();

		~FixedLayout() override;

		static Layout* Create() { return new FixedLayout(); }

		///  Prints out all parameters to logging file.
		///  Registered to OperationManager as Operation::printParameters
		void printParameters() const override;

		///  Creates a vertex type map.
		///
		///  @param  numVertices number of the vertices to have in the type map.
		void generateVertexTypeMap(int numVertices) override;

		///  Populates the starter map.
		///  Selects num_endogenously_active_neurons excitory neurons
		///  and converts them into starter vertices.
		///
		///  @param  numVertices number of vertices to have in the map.
		void initStarterMap(int numVertices) override;

		/// Load member variables from configuration file. Registered to OperationManager as Operation::loadParameters
		void loadParameters() override;

		/// Returns the type of synapse at the given coordinates
		/// @param    srcVertex  integer that points to a Neuron in the type map as a source.
		/// @param    destVertex integer that points to a Neuron in the type map as a destination.
		/// @return type of the synapse.
		edgeType edgType(int srcVertex, int destVertex) override;
};
