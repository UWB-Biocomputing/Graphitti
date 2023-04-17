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
 *     -# network wide spike count in 10ms bins.
 *
 * Hierarchical Data Format (HDF) is a set of file formats (HDF4, HDF5) designed
 * to store and organize large amounts of data.
 */
#pragma once

#if defined(HDF5)
   #include "H5Cpp.h"
   #include "IRecorder.h"
   #include "Model.h"
   #include <fstream>

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

   static IRecorder *Create()
   {
      return new Hdf5Recorder();
   }

   /// Initialize data
   /// @param[in] stateOutputFileName File name to save histories
   virtual void init() override;

   /// Init radii and rates history matrices with default values
   virtual void initDefaultValues() override;

   /// Init radii and rates history matrices with current radii and rates
   virtual void initValues() override;

   /// Get the current radii and rates vlaues
   virtual void getValues() override;

   /// Terminate process
   virtual void term() override;

   /// Compile history information in every epoch
   /// @param[in] neurons   The entire list of neurons.
   virtual void compileHistories(AllVertices &neurons) override;

   /// Writes simulation results to an output destination.
   /// @param  neurons the Neuron list to search from.
   virtual void saveSimData(const AllVertices &neurons) override;

   /// Prints out all parameters to logging file.
   /// Registered to OperationManager as Operation::printParameters
   virtual void printParameters() override;

protected:
   virtual void initDataSet();

   // Populates Starter neuron matrix based with boolean values based on starterMap state
   ///@param[in] matrix  starter neuron matrix
   ///@param starterMap  Bool vector to reference neuron matrix location from.
   virtual void getStarterNeuronMatrix(VectorMatrix &matrix,
                                       const std::vector<bool> &starterMap) override;

   /// hdf5 file identifier
   H5File resultOut_;

   /// hdf5 file dataset
   //static or simple dataset variables
   DataSet dataSetXloc_;
   DataSet dataSetYloc_;
   DataSet dataSetNeuronTypes_;
   DataSet dataSetNeuronThresh_;
   DataSet dataSetStarterNeurons_;
   DataSet dataSetTsim_;
   DataSet dataSetSimulationEndTime_;
   DataSet dataSetProbedNeurons_;

   //extentable dataset variable
   DataSet dataSetSpikesHist_;
   DataSet dataSetSpikesProbedNeurons_;

   /// Keep track of where we are in incrementally writing spikes
   hsize_t *offsetSpikesProbedNeurons_;

   /// spikes history - history of accumulated spikes count of all neurons (10 ms bin)
   int *spikesHistory_;

   /// track spikes count of probed neurons
   vector<uint64_t> *spikesProbedNeurons_;
};

#endif   // HDF5
