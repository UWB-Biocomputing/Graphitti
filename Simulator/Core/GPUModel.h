/**
 * @file GPUModel.h
 *
 * @ingroup Simulator/Core
 * 
 * @brief Implementation of Model for the graph-based networks.
 *
 * The Model class maintains and manages classes of objects that make up
 * essential components of the graph network.
 *    -# AllVertices: A class to define a list of particular type of vertices.
 *    -# AllEdges: A class to define a list of particular type of edges.
 *    -# Connections: A class to define connections of the graph network.
 *    -# Layout: A class to define vertices' layout information in the network.
 *
 * Edges in the edge map are located at the coordinates of the vertex
 * from which they receive output.
 *
 * The model runs on multi-threaded on a GPU.
 *
 */

#pragma once

#include "AllEdges.h"
#include "AllSpikingNeurons.h"
#include "AllSpikingSynapses.h"
#include "AllVertices.h"
#include "OperationManager.h"

#ifdef VALIDATION_MODE
   #include <fstream>
   #include <iostream>
#endif   // VALIDATION_MODE

#ifdef __CUDACC__
   #include "Book.h"
#endif

/************************************************
 * @name Inline functions for handling performance recording
 ***********************************************/
///@{
#if defined(PERFORMANCE_METRICS) && defined(__CUDACC__)
extern float g_time;
extern cudaEvent_t start, stop;

inline void cudaStartTimer()
{
   cudaEventRecord(start, 0);
};

//*! Increment elapsed time in seconds
inline void cudaLapTime(double &t_event)
{
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&g_time, start, stop);
   // The CUDA functions return time in milliseconds
   t_event += g_time / 1000.0;
};
#endif   // PERFORMANCE_METRICS
///@}

class AllEdges;

class GPUModel : public Model {
   friend class GpuSInputPoisson;

public:
   GPUModel();

   virtual ~GPUModel() = default;

   /// Set up model state, if anym for a specific simulation run.
   virtual void setupSim() override;

   /// Performs any finalization tasks on network following a simulation.
   virtual void finish() override;

   /// Advances network state one simulation step.
   virtual void advance() override;

   /// Modifies connections between vertices based on current state of the network and behavior
   /// over the past epoch. Should be called once every epoch.
   virtual void updateConnections() override;

   /// Copies neuron and synapse data from CPU to GPU memory.
   /// TODO: Refactor this. Currently, GPUModel handles low-level memory transfer for vertices and edges.
   ///       Consider moving this responsibility to a more appropriate class, such as a dedicated memory manager
   ///       or the OperationManager, to better separate concerns and keep the model focused on high-level coordination.
   virtual void copyCPUtoGPU() override;

   // GPUModel itself does not have anything to be copied back, this function is a
   // dummy function just to make GPUModel non virtual
   virtual void copyGPUtoCPU() override
   {
   }

   /// Print out EdgeProps on the GPU.
   void printGPUEdgesPropsModel() const;

   /// Getter for edge (synapse) structures in device memory
   AllEdgesDeviceProperties *&getAllEdgesDevice();

   /// Getter for vertex (neuron) structures in device memory
   AllVerticesDeviceProperties *&getAllVerticesDevice();

protected:
   /// Allocates  and initializes memories on CUDA device.
   void allocDeviceStruct();

   /// Deallocates device memories.
   virtual void deleteDeviceStruct();

   /// Pointer to device random noise array.
   float *randNoise_d;

#if defined(USE_GPU)
   /// Pointer to edge index map in device memory.
   EdgeIndexMapDevice *edgeIndexMapDevice_;
#endif   // defined(USE_GPU)

   /// edge structures in device memory.
   AllEdgesDeviceProperties *allEdgesDevice_;

   /// vertex structure in device memory.
   AllVerticesDeviceProperties *allVerticesDevice_;

private:
   void allocEdgeIndexMap(int count);

private:
   void updateHistory();

   // TODO
   void eraseEdge(AllEdges &edges, int vertexIndex, int edgeIndex);

   // TODO
   void addEdge(AllEdges &edges, edgeType type, int srcVertex, int destVertex, Coordinate &source,
                Coordinate &dest, BGFLOAT deltaT);

   // TODO
   void createEdge(AllEdges &edges, int vertexIndex, int edgeIndex, Coordinate source,
                   Coordinate dest, BGFLOAT deltaT, edgeType type);
};

#if defined(__CUDACC__)
extern "C" {
void normalMTGPU(float *randNoise_d);
void initMTGPU(unsigned int seed, unsigned int blocks, unsigned int threads, unsigned int nPerRng,
               unsigned int mt_rng_count);
}
#endif
