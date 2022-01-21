/**
 * @file All911Vertices.h
 * 
 * @ingroup Simulator/Vertices/NG911
 *
 * @brief A container of all 911 vertex data
 *
 * A container of all vertex data.
 *
 * The data-centric class uses a container structure for all vertices.
 *
 * The container holds vertex parameters for all vertices.
 * Each kind of vertex parameter is stored in a 1D array, of which length
 * is number of all vertices. Each array of a vertex parameter is pointed to by a
 * corresponding member variable of the vertex parameter in the class.
 *
 */
#pragma once

#include "AllVertices.h"
#include "Global.h"

// Class to hold all data necessary for all the Vertices.
class All911Vertices : public AllVertices {
	public:
		All911Vertices();

		~All911Vertices() override;

		///  Creates an instance of the class.
		///
		///  @return Reference to the instance of the class.
		static AllVertices* Create() { return new All911Vertices(); }

		///  Setup the internal structure of the class.
		///  Allocate memories to store all vertices' states.
		void setupVertices() override;

		///  Creates all the Vertices and assigns initial data for them.
		///
		///  @param  layout      Layout information of the network.
		void createAllVertices(Layout* layout) override;

		///  Load member variables from configuration file.
		///  Registered to OperationManager as Operation::loadParameters
		void loadParameters() override;

		///  Prints out all parameters of the vertices to logging file.
		///  Registered to OperationManager as Operation::printParameters
		void printParameters() const override;

		///  Outputs state of the vertex chosen as a string.
		///
		///  @param  index   index of the vertex (in vertices) to output info from.
		///  @return the complete state of the vertex.
		std::string toString(const int index) const override;

	private:
		/// number of callers
		int* callNum_;

		/// Min/max values of CallNum.
		int callNumRange_[2];

		/// Number of dispatchers per PSAP calculated (with randomness) based on population
		int* dispNum_;

		/// Scaling factor for number of dispatchers in a PSAP
		BGFLOAT dispNumScale_;

		/// Number of responders per Responder node calculated (with randomness) based on population
		int* respNum_;

		/// Scaling factor for number of responders in a Responder node
		BGFLOAT respNumScale_;

#ifdef __CUDACC__
   // GPU functionality for 911 simulation is unimplemented.
   // These signatures are required to make the class non-abstract
   public:
       virtual void allocNeuronDeviceStruct(void** allVerticesDevice) {};
       virtual void deleteNeuronDeviceStruct(void* allVerticesDevice) {};
       virtual void copyNeuronHostToDevice(void* allVerticesDevice) {};
       virtual void copyNeuronDeviceToHost(void* allVerticesDevice) {};
       virtual void advanceVertices(AllEdges &edges, void* allVerticesDevice, void* allEdgesDevice, float* randNoise, EdgeIndexMap* edgeIndexMapDevice) {};
       virtual void setAdvanceVerticesDeviceParams(AllEdges &edges) {};
#else  // !defined(__CUDACC__)
	public:
		///  Update internal state of the indexed Vertex (called by every simulation step).
		///  Notify outgoing edges if vertex has fired.
		///
		///  @param  edges         The Edge list to search from.
		///  @param  edgeIndexMap  Reference to the EdgeIndexMap.
		void advanceVertices(AllEdges& edges, const EdgeIndexMap* edgeIndexMap) override;

	protected:
		void advanceVertex(const int index);


#endif // defined(__CUDACC__)
};
