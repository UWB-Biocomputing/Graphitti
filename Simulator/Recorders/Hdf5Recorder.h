#pragma once

#if defined(HDF5)
   #include "H5Cpp.h"
   #include "Model.h"
   #include "Recorder.h"
   #include <fstream>
   #include <vector>

   #ifndef H5_NO_NAMESPACE
using namespace H5;
   #endif

   #ifdef SINGLEPRECISION
      #define H5_FLOAT PredType::NATIVE_FLOAT
   #else
      #define H5_FLOAT PredType::NATIVE_DOUBLE
   #endif

using std::string;
using std::vector;

class Hdf5Recorder : public Recorder {
public:
   Hdf5Recorder();

   static Recorder* Create()  {
      return new Hdf5Recorder();
   }

   // Other member functions...
   /// Initialize data
   /// @param[in] stateOutputFileName File name to save histories
   virtual void init() override {}

   /// Init radii and rates history matrices with default values
   virtual void initDefaultValues() override {}

   /// Init radii and rates history matrices with current radii and rates
   virtual void initValues() override {}

   /// Get the current radii and rates vlaues
   virtual void getValues() override {}

   /// Terminate process
   virtual void term() override {}

   // TODO: No parameters needed (AllVertices &vertices)
   /// Compile/capture variable history information in every epoch
   virtual void compileHistories(AllVertices &neurons) override {}

   // TODO: No parameters needed (AllVertices &vertices)
   /// Writes simulation results to an output destination.
   virtual void saveSimData(const AllVertices &neurons) override {}

   /// Prints out all parameters to logging file.
   /// Registered to OperationManager as Operation::printParameters
   virtual void printParameters() override {}

   /// Receives a recorded variable entity from the variable owner class
   /// used when the return type from recordable variable is supported by Recorder
   /**
   * @brief Registers a single instance of a class derived from RecordableBase.
   * @param varName Name of the recorded variable.
   * @param recordVar Reference to the recorded variable.
   * @param variableType Type of the recorded variable.
   */
   virtual void registerVariable(const string &varName, RecordableBase &recordVar,
                                 UpdatedType variableType) override {}

   /// Register a vector of instance of a class derived from RecordableBase.
   virtual void registerVariable(const string &varName, vector<RecordableBase *> &recordVars,
                                 UpdatedType variableType) override {}

private:
   virtual void initDataSet() {}

   /// Populates Starter neuron matrix based with boolean values based on starterMap state
   ///@param[in] matrix  starter neuron matrix
   ///@param starterMap  Bool vector to reference neuron matrix location from.
   virtual void getStarterNeuronMatrix(VectorMatrix &matrix,
                                       const vector<bool> &starterMap) override {}

   // Member variables for HDF5 datasets
   H5File* resultOut_;
   DataSet dataSetXloc_;
   DataSet dataSetYloc_;
   DataSet dataSetNeuronTypes_;
   DataSet dataSetNeuronThresh_;
   DataSet dataSetStarterNeurons_;
   DataSet dataSetTsim_;
   DataSet dataSetSimulationEndTime_;
   DataSet dataSetProbedNeurons_;
   DataSet dataSetSpikesHist_;
   DataSet dataSetSpikesProbedNeurons_;

   // HDF5 dataset names
   const H5std_string nameSpikesHist = "spikesHistory";
   const H5std_string nameXloc = "xloc";
   const H5std_string nameYloc = "yloc";
   const H5std_string nameNeuronTypes = "neuronTypes";
   const H5std_string nameNeuronThresh = "neuronThresh";
   const H5std_string nameStarterNeurons = "starterNeurons";
   const H5std_string nameTsim = "Tsim";
   const H5std_string nameSimulationEndTime = "simulationEndTime";
   const H5std_string nameSpikesProbedNeurons = "spikesProbedNeurons";
   const H5std_string nameAttrPNUnit = "attrPNUint";
   const H5std_string nameProbedNeurons = "probedNeurons";

   // Keep track of where we are in incrementally writing spikes
   vector<hsize_t> offsetSpikesProbedNeurons_;
   // spikes history - history of accumulated spikes count of all neurons (10 ms bin)
   vector<int> spikesHistory_;
   // track spikes count of probed neurons
   vector<vector<uint64_t>> spikesProbedNeurons_;

   // Other member functions...
   
};

#endif // HDF5
