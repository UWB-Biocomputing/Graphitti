/**
 * @file Hdf5Recorder.cpp
 * 
 * @ingroup Simulator/Recorders
 *
 * @brief An implementation for recording spikes history on hdf5 file
 */

#include "Hdf5Recorder.h"
#include "AllIFNeurons.h"   // TODO: remove LIF model specific code
#include "OperationManager.h"
#include "ParameterManager.h"

#if defined(HDF5)

// hdf5 dataset name
const H5std_string nameBurstHist("burstinessHist");
const H5std_string nameSpikesHist("spikesHistory");
const H5std_string nameXloc("xloc");
const H5std_string nameYloc("yloc");
const H5std_string nameNeuronTypes("neuronTypes");
const H5std_string nameNeuronThresh("neuronThresh");
const H5std_string nameStarterNeurons("starterNeurons");
const H5std_string nameTsim("Tsim");
const H5std_string nameSimulationEndTime("simulationEndTime");
const H5std_string nameSpikesProbedNeurons("spikesProbedNeurons");
const H5std_string nameAttrPNUnit("attrPNUint");
const H5std_string nameProbedNeurons("probedNeurons");

/// The constructor and destructor
Hdf5Recorder::Hdf5Recorder() : offsetSpikesProbedNeurons_(nullptr), spikesProbedNeurons_(nullptr)
{
   ParameterManager::getInstance().getStringByXpath(
      "//RecorderParams/RecorderFiles/resultFileName/text()", resultFileName_);

   function<void()> printParametersFunc = std::bind(&Hdf5Recorder::printParameters, this);
   OperationManager::getInstance().registerOperation(Operations::printParameters,
                                                     printParametersFunc);

   fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));
}


/// Initialize data
/// Create a new hdf5 file with default properties.
/// @param[in] stateOutputFileName	File name to save histories
void Hdf5Recorder::init()
{
   // Before trying to create H5File, use ofstream to confirm ability to create and write file.
   // TODO: Log error using LOG4CPLUS for workbench
   //       For the time being, we are terminating the program when we can't open a file for writing.
   ofstream testFileWrite;
   testFileWrite.open(resultFileName_.c_str());
   if (!testFileWrite.is_open()) {
      perror("Error opening output file for writing ");
      exit(EXIT_FAILURE);
   }
   testFileWrite.close();

   try {
      // create a new file using the default property lists
      resultOut_ = new H5File(resultFileName_, H5F_ACC_TRUNC);
      initDataSet();
   }

   // catch failure caused by the H5File operations
   catch (FileIException error) {
      error.printErrorStack();
      return;
   }

   // catch failure caused by the DataSet operations
   catch (DataSetIException error) {
      error.printErrorStack();
      return;
   }

   // catch failure caused by the DataSpace operations
   catch (DataSpaceIException error) {
      error.printErrorStack();
      return;
   }

   // catch failure caused by the DataType operations
   catch (DataTypeIException error) {
      error.printErrorStack();
      return;
   }
}

///  Create data spaces and data sets of the hdf5 for recording histories.
void Hdf5Recorder::initDataSet()
{
   Simulator &simulator = Simulator::getInstance();

   // create the data space & dataset for burstiness history
   hsize_t dims[2];
   dims[0] = static_cast<hsize_t>(simulator.getEpochDuration() * simulator.getNumEpochs());
   DataSpace dsBurstHist(1, dims);
   dataSetBurstHist_
      = new DataSet(resultOut_->createDataSet(nameBurstHist, PredType::NATIVE_INT, dsBurstHist));

   // create the data space & dataset for spikes history
   dims[0] = static_cast<hsize_t>(simulator.getEpochDuration() * simulator.getNumEpochs() * 100);
   DataSpace dsSpikesHist(1, dims);
   dataSetSpikesHist_
      = new DataSet(resultOut_->createDataSet(nameSpikesHist, PredType::NATIVE_INT, dsSpikesHist));

   // create the data space & dataset for xloc & yloc
   dims[0] = static_cast<hsize_t>(simulator.getTotalVertices());
   DataSpace dsXYloc(1, dims);
   dataSetXloc_ = new DataSet(resultOut_->createDataSet(nameXloc, PredType::NATIVE_INT, dsXYloc));
   dataSetYloc_ = new DataSet(resultOut_->createDataSet(nameYloc, PredType::NATIVE_INT, dsXYloc));

   // create the data space & dataset for neuron types
   dims[0] = static_cast<hsize_t>(simulator.getTotalVertices());
   DataSpace dsNeuronTypes(1, dims);
   dataSetNeuronTypes_ = new DataSet(
      resultOut_->createDataSet(nameNeuronTypes, PredType::NATIVE_INT, dsNeuronTypes));

   // create the data space & dataset for neuron threshold
   dims[0] = static_cast<hsize_t>(simulator.getTotalVertices());
   DataSpace dsNeuronThresh(1, dims);
   dataSetNeuronThresh_
      = new DataSet(resultOut_->createDataSet(nameNeuronThresh, H5_FLOAT, dsNeuronThresh));

   // create the data space & dataset for simulation step duration
   dims[0] = static_cast<hsize_t>(1);
   DataSpace dsTsim(1, dims);
   dataSetTsim_ = new DataSet(resultOut_->createDataSet(nameTsim, H5_FLOAT, dsTsim));

   // create the data space & dataset for simulation end time
   dims[0] = static_cast<hsize_t>(1);
   DataSpace dsSimulationEndTime(1, dims);
   dataSetSimulationEndTime_ = new DataSet(
      resultOut_->createDataSet(nameSimulationEndTime, H5_FLOAT, dsSimulationEndTime));

   // Get model instance
   Model *model = simulator.getModel();

   // Set up probed neurons so that they can be written incrementally
   if (model->getLayout()->probedNeuronList_.size() > 0) {
      // create the data space & dataset for probed neurons
      dims[0] = static_cast<hsize_t>(model->getLayout()->probedNeuronList_.size());
      DataSpace dsProbedNeurons(1, dims);
      dataSetProbedNeurons_ = new DataSet(
         resultOut_->createDataSet(nameProbedNeurons, PredType::NATIVE_INT, dsProbedNeurons));

      // create the data space & dataset for spikes of probed neurons

      // the data space with unlimited dimensions
      hsize_t maxdims[2];
      maxdims[0] = H5S_UNLIMITED;
      maxdims[1] = static_cast<hsize_t>(model->getLayout()->probedNeuronList_.size());

      // dataset dimensions at creation
      dims[0] = static_cast<hsize_t>(1);
      dims[1] = static_cast<hsize_t>(model->getLayout()->probedNeuronList_.size());
      DataSpace dsSpikesProbedNeurons(2, dims, maxdims);

      // set fill value for the dataset
      DSetCreatPropList cparms;
      uint64_t fill_val = 0;
      cparms.setFillValue(PredType::NATIVE_UINT64, &fill_val);

      // modify dataset creation properties, enable chunking
      hsize_t chunk_dims[2];
      chunk_dims[0] = static_cast<hsize_t>(100);
      chunk_dims[1] = static_cast<hsize_t>(model->getLayout()->probedNeuronList_.size());
      cparms.setChunk(2, chunk_dims);

      dataSetSpikesProbedNeurons_ = new DataSet(resultOut_->createDataSet(
         nameSpikesProbedNeurons, PredType::NATIVE_UINT64, dsSpikesProbedNeurons, cparms));
   }

   // allocate and initialize data memories
   burstinessHist_ = new int[static_cast<int>(simulator.getEpochDuration())];
   spikesHistory_ = new int[static_cast<int>(simulator.getEpochDuration() * 100)];
   memset(burstinessHist_, 0, static_cast<int>(simulator.getEpochDuration() * sizeof(int)));
   memset(spikesHistory_, 0, static_cast<int>(simulator.getEpochDuration() * 100 * sizeof(int)));

   // create the data space & dataset for spikes history of probed neurons
   if (model->getLayout()->probedNeuronList_.size() > 0) {
      // allocate data for spikesProbedNeurons
      spikesProbedNeurons_ = new vector<uint64_t>[model->getLayout()->probedNeuronList_.size()];

      // allocate and initialize memory to save offsets of what's been written
      offsetSpikesProbedNeurons_ = new hsize_t[model->getLayout()->probedNeuronList_.size()];
      memset(offsetSpikesProbedNeurons_, 0,
             static_cast<int>(model->getLayout()->probedNeuronList_.size() * sizeof(hsize_t)));
   }
}

// TODO: for these empty methods, should anything happen? Should they never be
// TODO: called?
/// Init history matrices with default values
void Hdf5Recorder::initDefaultValues()
{
}

/// Init history matrices with current radii and rates
void Hdf5Recorder::initValues()
{
}

/// Get the current values
void Hdf5Recorder::getValues()
{
}

/// Terminate process
void Hdf5Recorder::term()
{
   // deallocate all objects
   delete[] burstinessHist_;
   delete[] spikesHistory_;

   delete dataSetBurstHist_;
   delete dataSetSpikesHist_;

   Model *model = Simulator::getInstance().getModel();

   if (model->getLayout()->probedNeuronList_.size() > 0) {
      delete dataSetProbedNeurons_;
      delete dataSetSpikesProbedNeurons_;

      delete[] spikesProbedNeurons_;
      delete[] offsetSpikesProbedNeurons_;
   }

   delete resultOut_;
}

/// Compile history information in every epoch.
/// @param[in] neurons   The entire list of neurons.
void Hdf5Recorder::compileHistories(AllVertices &vertices)
{
   AllSpikingNeurons &spNeurons = dynamic_cast<AllSpikingNeurons &>(vertices);
   Simulator &simulator = Simulator::getInstance();

   int maxSpikes = (int)((simulator.getEpochDuration() * simulator.getMaxFiringRate()));

   unsigned int iProbe = 0;   // index into the probedNeuronsLayout vector
   bool fProbe = false;

   Model *model = simulator.getModel();

   // output spikes: iterate over each neuron
   for (int iVertex = 0; iVertex < spNeurons.vertexEvents_.size(); iVertex++) {
      // true if this is a probed neuron
      fProbe = ((iProbe < model->getLayout()->probedNeuronList_.size())
                && (iVertex == model->getLayout()->probedNeuronList_[iProbe]));

      // iterate over each spike that neuron produced
      for (int eventIterator = 0;
           eventIterator < spNeurons.vertexEvents_[iVertex].getNumEventsInEpoch();
           eventIterator++) {
         // Single precision (float) gives you 23 bits of significand, 8 bits of exponent,
         // and 1 sign bit. Double precision (double) gives you 52 bits of significand,
         // 11 bits of exponent, and 1 sign bit.
         // Therefore, single precision can only handle 2^23 = 8,388,608 simulation steps
         // or 8 epochs (1 epoch = 100s, 1 simulation step = 0.1ms).

         // if (idxSp >= maxSpikes) idxSp = 0;
         //  compile network wide burstiness index data in 1s bins
         int idx1
            = static_cast<int>(static_cast<double>(spNeurons.vertexEvents_[iVertex][eventIterator])
                                  * simulator.getDeltaT()
                               - ((simulator.getCurrentStep() - 1) * simulator.getEpochDuration()));
         // make sure idx1 is a valid index of burstinessHist_
         assert(idx1 >= 0 && idx1 < simulator.getEpochDuration());
         burstinessHist_[idx1] = burstinessHist_[idx1] + 1.0;

         // compile network wide spike count in 10ms bins
         int idx2 = static_cast<int>(
            static_cast<double>(spNeurons.vertexEvents_[iVertex][eventIterator])
               * simulator.getDeltaT() * 100
            - ((simulator.getCurrentStep() - 1) * simulator.getEpochDuration() * 100));
         // make sure idx2 is a valid index of spikesHistory_
         assert(idx2 >= 0 && idx2 < (simulator.getEpochDuration() * 100));
         spikesHistory_[idx2] = spikesHistory_[idx2] + 1.0;

         // compile spikes time of the probed neuron (append spikes time)
         if (fProbe) {
            spikesProbedNeurons_[iProbe].insert(spikesProbedNeurons_[iProbe].end(),
                                                spNeurons.vertexEvents_[iVertex][eventIterator]);
         }
      }

      if (fProbe) {
         iProbe++;
      }
   }

   // clear spike count for all neurons (we've captured their spike information)
   spNeurons.clearSpikeCounts();

   try {
      // write burstiness index
      hsize_t offset[2], count[2];
      hsize_t dimsm[2];
      DataSpace *dataspace;
      DataSpace *memspace;

      offset[0] = (simulator.getCurrentStep() - 1) * simulator.getEpochDuration();
      count[0] = simulator.getEpochDuration();
      dimsm[0] = simulator.getEpochDuration();
      memspace = new DataSpace(1, dimsm, nullptr);
      dataspace = new DataSpace(dataSetBurstHist_->getSpace());
      dataspace->selectHyperslab(H5S_SELECT_SET, count, offset);
      dataSetBurstHist_->write(burstinessHist_, PredType::NATIVE_INT, *memspace, *dataspace);
      memset(burstinessHist_, 0, static_cast<int>(simulator.getEpochDuration() * sizeof(int)));
      delete dataspace;
      delete memspace;

      // write network wide spike count in 10ms bins
      offset[0] = (simulator.getCurrentStep() - 1) * simulator.getEpochDuration() * 100;
      count[0] = simulator.getEpochDuration() * 100;
      dimsm[0] = simulator.getEpochDuration() * 100;
      memspace = new DataSpace(1, dimsm, nullptr);
      dataspace = new DataSpace(dataSetSpikesHist_->getSpace());
      dataspace->selectHyperslab(H5S_SELECT_SET, count, offset);
      dataSetSpikesHist_->write(spikesHistory_, PredType::NATIVE_INT, *memspace, *dataspace);
      memset(spikesHistory_, 0, static_cast<int>(simulator.getEpochDuration() * 100 * sizeof(int)));
      delete dataspace;
      delete memspace;

      // write spikes data of probed neurons
      if (model->getLayout()->probedNeuronList_.size() > 0) {
         unsigned int max_size = 0;
         // iterate over each neuron to find the maximum number of spikes for
         // this epoch
         for (unsigned int i = 0; i < model->getLayout()->probedNeuronList_.size(); i++) {
            unsigned int size = spikesProbedNeurons_[i].size() + offsetSpikesProbedNeurons_[i];
            max_size = (max_size > size) ? max_size : size;
         }
         // dataset dimensions
         dimsm[0] = static_cast<hsize_t>(max_size);
         dimsm[1] = static_cast<hsize_t>(model->getLayout()->probedNeuronList_.size());

         // extend the dataset
         dataSetSpikesProbedNeurons_->extend(dimsm);
         dataspace = new DataSpace(dataSetSpikesProbedNeurons_->getSpace());

         // write it! Iterate over each neuron's spike data.
         for (unsigned int i = 0; i < model->getLayout()->probedNeuronList_.size(); i++) {
            dimsm[0] = spikesProbedNeurons_[i].size();
            dimsm[1] = 1;
            memspace = new DataSpace(2, dimsm, nullptr);

            offset[0] = offsetSpikesProbedNeurons_[i];
            offset[1] = i;
            count[0] = spikesProbedNeurons_[i].size();
            count[1] = 1;
            dataspace->selectHyperslab(H5S_SELECT_SET, count, offset);
            offsetSpikesProbedNeurons_[i] += spikesProbedNeurons_[i].size();

            dataSetSpikesProbedNeurons_->write(
               static_cast<uint64_t *>(&(spikesProbedNeurons_[i][0])), PredType::NATIVE_UINT64,
               *memspace, *dataspace);

            // clear the probed spike data
            spikesProbedNeurons_[i].clear();
            delete memspace;
         }

         delete dataspace;
      }
   }

   // catch failure caused by the H5File operations
   catch (FileIException error) {
      error.printErrorStack();
      return;
   }

   // catch failure caused by the DataSet operations
   catch (DataSetIException error) {
      error.printErrorStack();
      return;
   }

   // catch failure caused by the DataSpace operations
   catch (DataSpaceIException error) {
      error.printErrorStack();
      return;
   }

   // catch failure caused by the DataType operations
   catch (DataTypeIException error) {
      error.printErrorStack();
      return;
   }
}

/// Writes simulation results to an output destination.
///
/// @param  vertices the AllVertices object.
void Hdf5Recorder::saveSimData(const AllVertices &vertices)
{
   Simulator &simulator = Simulator::getInstance();
   Model *model = simulator.getModel();

   try {
      // create Neuron Types matrix
      VectorMatrix neuronTypes(MATRIX_TYPE, MATRIX_INIT, 1, simulator.getTotalVertices(), EXC);
      for (int i = 0; i < simulator.getTotalVertices(); i++) {
         neuronTypes[i] = model->getLayout()->vertexTypeMap_[i];
      }

      // create neuron threshold matrix
      VectorMatrix neuronThresh(MATRIX_TYPE, MATRIX_INIT, 1, simulator.getTotalVertices(), 0);
      for (int i = 0; i < simulator.getTotalVertices(); i++) {
         neuronThresh[i] = dynamic_cast<const AllIFNeurons &>(vertices).Vthresh_[i];
      }

      // Write the neuron location matrices
      int *iXloc = new int[simulator.getTotalVertices()];
      int *iYloc = new int[simulator.getTotalVertices()];
      for (int i = 0; i < simulator.getTotalVertices(); i++) {
         // convert VectorMatrix to int array
         iXloc[i] = (model->getLayout()->xloc_)[i];
         iYloc[i] = (model->getLayout()->yloc_)[i];
      }
      dataSetXloc_->write(iXloc, PredType::NATIVE_INT);
      dataSetYloc_->write(iYloc, PredType::NATIVE_INT);
      delete[] iXloc;
      delete[] iYloc;

      int *iNeuronTypes = new int[simulator.getTotalVertices()];
      for (int i = 0; i < simulator.getTotalVertices(); i++) {
         iNeuronTypes[i] = neuronTypes[i];
      }
      dataSetNeuronTypes_->write(iNeuronTypes, PredType::NATIVE_INT);
      delete[] iNeuronTypes;

      int num_starter_neurons = static_cast<int>(model->getLayout()->numEndogenouslyActiveNeurons_);
      if (num_starter_neurons > 0) {
         VectorMatrix starterNeurons(MATRIX_TYPE, MATRIX_INIT, 1, num_starter_neurons);
         getStarterNeuronMatrix(starterNeurons, model->getLayout()->starterMap_);

         // create the data space & dataset for starter neurons
         hsize_t dims[2];
         dims[0] = static_cast<hsize_t>(starterNeurons.Size());
         DataSpace dsStarterNeurons(1, dims);
         dataSetStarterNeurons_ = new DataSet(
            resultOut_->createDataSet(nameStarterNeurons, PredType::NATIVE_INT, dsStarterNeurons));

         int *iStarterNeurons = new int[starterNeurons.Size()];
         for (int i = 0; i < starterNeurons.Size(); i++) {
            iStarterNeurons[i] = starterNeurons[i];
         }
         dataSetStarterNeurons_->write(iStarterNeurons, PredType::NATIVE_INT);
         delete[] iStarterNeurons;
         delete dataSetStarterNeurons_;
      }

      // Finalize probed neurons' spikes dataset
      if (model->getLayout()->probedNeuronList_.size() > 0) {
         // create the data space & dataset for probed neurons
         hsize_t dims[2];

         int *iProbedNeurons = new int[model->getLayout()->probedNeuronList_.size()];
         for (unsigned int i = 0; i < model->getLayout()->probedNeuronList_.size(); i++) {
            iProbedNeurons[i] = model->getLayout()->probedNeuronList_[i];
         }
         dataSetProbedNeurons_->write(iProbedNeurons, PredType::NATIVE_INT);
         delete[] iProbedNeurons;

         // Create the data space for the attribute (unit of the spikes of probed neurons in second).
         dims[0] = 1;
         DataSpace dsAttrPNUnit(1, dims);

         // Create a dataset attribute.
         Attribute attribute = dataSetSpikesProbedNeurons_->createAttribute(
            nameAttrPNUnit, H5_FLOAT, dsAttrPNUnit, PropList::DEFAULT);

         // Write the attribute data.
         float deltaT = Simulator::getInstance().getDeltaT();
         attribute.write(H5_FLOAT, &deltaT);
      }

      // Write neuron thresold
      BGFLOAT *fNeuronThresh = new BGFLOAT[simulator.getTotalVertices()];
      for (int i = 0; i < simulator.getTotalVertices(); i++) {
         fNeuronThresh[i] = neuronThresh[i];
      }
      dataSetNeuronThresh_->write(fNeuronThresh, H5_FLOAT);
      delete[] fNeuronThresh;

      // write time between growth cycles
      BGFLOAT epochDuration = simulator.getEpochDuration();
      dataSetTsim_->write(&epochDuration, H5_FLOAT);
      delete dataSetTsim_;

      // write simulation end time
      BGFLOAT endTime = g_simulationStep * simulator.getDeltaT();
      dataSetSimulationEndTime_->write(&endTime, H5_FLOAT);
      delete dataSetSimulationEndTime_;
   }

   // catch failure caused by the DataSet operations
   catch (DataSetIException error) {
      error.printErrorStack();
      return;
   }

   // catch failure caused by the DataSpace operations
   catch (DataSpaceIException error) {
      error.printErrorStack();
      return;
   }
}

// TODO: this seems to be duplicated in multiple Recorder classes
///  Get starter Neuron matrix.
///
///  @param  matrix      Starter Neuron matrix.
///  @param  startermap Bool map to reference neuron matrix location from.
void Hdf5Recorder::getStarterNeuronMatrix(VectorMatrix &matrix, const std::vector<bool> &starterMap)
{
   int cur = 0;
   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      if (starterMap[i]) {
         matrix[cur] = i;
         cur++;
      }
   }
}

/**
 *  Prints out all parameters to logging file.
 *  Registered to OperationManager as Operation::printParameters
 */
void Hdf5Recorder::printParameters()
{
   LOG4CPLUS_DEBUG(fileLogger_, "\nHdf5Recorder Parameters"
                                   << endl
                                   << "\tResult file path: " << resultFileName_ << endl);
}

#endif   // HDF5
