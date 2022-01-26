/**
 * @file Connections911.h
 *
 * @ingroup Simulator/Connections/NG911
 * 
 * @brief The model of the static network
 *
 */

#pragma once

#include <vector>
#include "Connections.h"
#include "Global.h"
#include "Simulator.h"

class Connections911 : public Connections {
	public:
		Connections911();

		~Connections911() override;

		///  Creates an instance of the class.
		///
		///  @return Reference to the instance of the class.
		static Connections* Create() { return new Connections911(); }

		///  Setup the internal structure of the class (allocate memories and initialize them).
		///  Initialize the network characterized by parameters:
		///  number of maximum connections per vertex, connection radius threshold
		///
		///  @param  layout    Layout information of the network.
		///  @param  vertices  The Vertex list to search from.
		///  @param  edges     The edge list to search from.
		void setupConnections(Layout* layout, AllVertices* vertices, AllEdges* edges) override;

		/// Load member variables from configuration file.
		/// Registered to OperationManager as Operations::op::loadParameters
		void loadParameters() override;

		///  Prints out all parameters to logging file.
		///  Registered to OperationManager as Operation::printParameters
		void printParameters() const override;

		/// Records typeMap history for recorders
		vertexType* oldTypeMap_;

	private:
		/// number of maximum connections per vertex
		int connsPerVertex_;

		/// number of psaps to erase at the end of 1 epoch
		int psapsToErase_;

		/// number of responder nodes to erase at the end of 1 epoch
		int respsToErase_;

		struct ChangedEdge;

		// Edges that were added but later removed are still here
		std::vector<ChangedEdge> edgesAdded;

		// New edges = (old edges + edgesAdded) - edgesErased  <-- works
		// New edges = (old edges - edgesErased) + edgesAdded  <-- does not work
		std::vector<ChangedEdge> edgesErased;

		std::vector<int> verticesErased;

#ifndef __CUDACC__

	public:
		///  Update the connections status in every epoch.
		///  Uses the parent definition for __CUDACC__
		///
		///  @param  vertices The Vertex list to search from.
		///  @param  layout   Layout information of the vertex network.
		///  @return true if successful, false otherwise.
		bool updateConnections(AllVertices& vertices, Layout* layout) override;

		///  Returns the complete list of all deleted or added edges as a string.
		///  @return xml representation of all deleted or added edges
		std::string changedEdgesToXML(bool added);

		///  Returns the complete list of deleted vertices as a string.
		///  @return xml representation of all deleted vertices
		std::string erasedVerticesToXML();

	private:
		///  Randomly delete 1 PSAP and rewire all the edges around it.
		///
		///  @param  vertices  The Vertex list to search from.
		///  @param  layout   Layout information of the vertex network.
		///  @return true if successful, false otherwise.
		bool erasePSAP(AllVertices& vertices, Layout* layout);

		///  Randomly delete 1 RESP.
		///
		///  @param  vertices  The Vertex list to search from.
		///  @param  layout   Layout information of the vertex network.
		///  @return true if successful, false otherwise.
		bool eraseRESP(AllVertices& vertices, Layout* layout);

		struct ChangedEdge {
			int srcV;
			int destV;
			edgeType eType;
			std::string toString();
		};

#else
public:
   // Not Implemented; Placeholder for GPU build
   string erasedVerticesToXML() { return ""; };
   string changedEdgesToXML(bool added) { return ""; };

private:
   // Not Implemented; Placeholder for GPU build
   struct ChangedEdge {};

#endif

};
