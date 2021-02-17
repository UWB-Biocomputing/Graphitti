/**
 * @file Hdf5Recorder.h
 * 
 * @ingroup Simulator/Recorders
 *
 * @brief An implementation for recording spikes history on hdf5 file
 *
 * The Hdf5Recorder provides a mechanism for recording neuron's layout, spikes history,
 * and compile history information on hdf5 file:
 *     -# neuron's locations, and type map,
 *     -# individual neuron's spike rate in epochs,
 *     -# network wide burstiness index data in 1s bins,
 *     -# network wide spike count in 10ms bins.
 *
 * Hierarchical Data Format (HDF) is a set of file formats (HDF4, HDF5) designed 
 * to store and organize large amounts of data.
 */
#pragma once

#if defined(HDF5)

#include "IRecorder.h"
#include "Model.h"
#include "H5Cpp.h"

#ifndef H5_NO_NAMESPACE
using namespace H5;
#endif

#ifdef SINGLEPRECISION
#define H5_FLOAT PredType::NATIVE_FLOAT
#else
#define H5_FLOAT PredType::NATIVE_DOUBLE
#endif

class Hdf5Recorder : public IRecorder {
public:
   /// THe constructor and destructor
   Hdf5Recorder();

   ~Hdf5Recorder();

   static IRecorder* Create() { return new Hdf5Recorder(); }

   /// Initialize data
   ///
   /// @param[in] stateOutputFileName       File name to save histories
   virtual void init();

   /// Init radii and rates history matrices with default values
   virtual void initDefaultValues();

   /// Init radii and rates history matrices with current radii and rates
   virtual void initValues();

   /// Get the current radii and rates vlaues
   virtual void getValues();

   /// Terminate process
   virtual void term();

   /// Compile history information in every epoch
   ///
   /// @param[in] neurons   The entire list of neurons.
   virtual void compileHistories(IAllVertices &neurons);

   /// Writes simulation results to an output destination.
   ///
   /// @param  neurons the Neuron list to search from.
   virtual void saveSimData(const IAllVertices &neurons);
   
   /// Prints out all parameters to logging file.
   /// Registered to OperationManager as Operation::printParameters
   virtual void printParameters();

protected:
   virtual void initDataSet();

   void getStarterNeuronMatrix(VectorMatrix &matrix, const bool *starterMap);

   /// hdf5 file identifier
   H5File *stateOut_;

   /// hdf5 file dataset
   DataSet *dataSetBurstHist_;
   DataSet *dataSetSpikesHist_;

   DataSet *dataSetXloc_;
   DataSet *dataSetYloc_;
   DataSet *dataSetNeuronTypes_;
   DataSet *dataSetNeuronThresh_;
   DataSet *dataSetStarterNeurons_;
   DataSet *dataSetTsim_;
   DataSet *dataSetSimulationEndTime_;

   DataSet *dataSetSpikesProbedNeurons_;
   DataSet *dataSetProbedNeurons_;

   /// Keep track of where we are in incrementally writing spikes
   hsize_t* offsetSpikesProbedNeurons_;

   /// burstiness Histogram goes through the
   int* burstinessHist_;

   /// spikes history - history of accumulated spikes count of all neurons (10 ms bin)
   int *spikesHistory_;

   /// track spikes count of probed neurons
   vector<uint64_t> *spikesProbedNeurons_;
};

#endif //HDF5
