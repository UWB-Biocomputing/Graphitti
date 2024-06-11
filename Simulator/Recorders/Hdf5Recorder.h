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

    static Recorder* Create();

    // Other member functions...

private:
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

    // Logger and file name
    log4cplus::Logger fileLogger_;
    string resultFileName_;

    // Other member functions...
};

#endif // HDF5
