/*
 *      @file Hdf5Recorder.cpp
 *
 *      @brief An implementation for recording spikes history on hdf5 file
 */
//! An implementation for recording spikes history on hdf5 file

#include "Hdf5Recorder.h"

#include "AllIFNeurons.h"      // TODO: remove LIF model specific code
#include "OperationManager.h"

// hdf5 dataset name
const H5std_string  nameBurstHist("burstinessHist");
const H5std_string  nameSpikesHist("spikesHistory");

const H5std_string  nameXloc("xloc");
const H5std_string  nameYloc("yloc");
const H5std_string  nameNeuronTypes("neuronTypes");
const H5std_string  nameNeuronThresh("neuronThresh");
const H5std_string  nameStarterNeurons("starterNeurons");
const H5std_string  nameTsim("Tsim");
const H5std_string  nameSimulationEndTime("simulationEndTime");

const H5std_string  nameSpikesProbedNeurons("spikesProbedNeurons");
const H5std_string  nameAttrPNUnit("attrPNUint");
const H5std_string  nameProbedNeurons("probedNeurons");

//! The constructor and destructor
Hdf5Recorder::Hdf5Recorder() :
      offsetSpikesProbedNeurons_(NULL),
      spikesProbedNeurons_(NULL) {

   resultFileName_ = Simulator::getInstance().getResultFileName();

   function<void()> printParametersFunc = std::bind(&Hdf5Recorder::printParameters, this);
   OperationManager::getInstance().registerOperation(Operations::printParameters, printParametersFunc);

   fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));
}

Hdf5Recorder::~Hdf5Recorder()
{
}

/*
 * Initialize data
 * Create a new hdf5 file with default properties.
 *
 * @param[in] stateOutputFileName	File name to save histories
 */
void Hdf5Recorder::init()
{
    try
    {
       
        // create a new file using the default property lists
        stateOut_ = new H5File(resultFileName_, H5F_ACC_TRUNC );

        initDataSet();
    }
    
    // catch failure caused by the H5File operations
    catch( FileIException error )
    {
        error.printErrorStack();
        return;
    }

    // catch failure caused by the DataSet operations
    catch( DataSetIException error )
    {
        error.printErrorStack();
        return;
    }

    // catch failure caused by the DataSpace operations
    catch( DataSpaceIException error )
    {
        error.printErrorStack();
        return;
    }

    // catch failure caused by the DataType operations
    catch( DataTypeIException error )
    {
        error.printErrorStack();
        return;
    }
}

/*
 *  Create data spaces and data sets of the hdf5 for recording histories.
 */
void Hdf5Recorder::initDataSet()
{
   // create the data space & dataset for burstiness history
   hsize_t dims[2];
   dims[0] = static_cast<hsize_t>(Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getNumEpochs());
   DataSpace dsBurstHist(1, dims);
   dataSetBurstHist_ = new DataSet(stateOut_->createDataSet(nameBurstHist, PredType::NATIVE_INT, dsBurstHist));

   // create the data space & dataset for spikes history
   dims[0] = static_cast<hsize_t>(Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getNumEpochs() * 100);
   DataSpace dsSpikesHist(1, dims);
   dataSetSpikesHist_ = new DataSet(stateOut_->createDataSet(nameSpikesHist, PredType::NATIVE_INT, dsSpikesHist));

   // create the data space & dataset for xloc & yloc
   dims[0] = static_cast<hsize_t>(Simulator::getInstance().getTotalNeurons());
   DataSpace dsXYloc(1, dims);
   dataSetXloc_ = new DataSet(stateOut_->createDataSet(nameXloc, PredType::NATIVE_INT, dsXYloc));
   dataSetYloc_ = new DataSet(stateOut_->createDataSet(nameYloc, PredType::NATIVE_INT, dsXYloc));

   // create the data space & dataset for neuron types
   dims[0] = static_cast<hsize_t>(Simulator::getInstance().getTotalNeurons());
   DataSpace dsNeuronTypes(1, dims);
   dataSetNeuronTypes_ = new DataSet(stateOut_->createDataSet(nameNeuronTypes, PredType::NATIVE_INT, dsNeuronTypes));

   // create the data space & dataset for neuron threshold
   dims[0] = static_cast<hsize_t>(Simulator::getInstance().getTotalNeurons());
   DataSpace dsNeuronThresh(1, dims);
   dataSetNeuronThresh_ = new DataSet(stateOut_->createDataSet(nameNeuronThresh, H5_FLOAT, dsNeuronThresh));

   // create the data space & dataset for simulation step duration
   dims[0] = static_cast<hsize_t>(1);
   DataSpace dsTsim(1, dims);
   dataSetTsim_ = new DataSet(stateOut_->createDataSet(nameTsim, H5_FLOAT, dsTsim));

   // create the data space & dataset for simulation end time
   dims[0] = static_cast<hsize_t>(1);
   DataSpace dsSimulationEndTime(1, dims);
   dataSetSimulationEndTime_ = new DataSet(stateOut_->createDataSet(nameSimulationEndTime, H5_FLOAT, dsSimulationEndTime));

   // Get model instance
   shared_ptr<Model> model = Simulator::getInstance().getModel();

   // Set up probed neurons so that they can be written incrementally
   if (model->getLayout()->probedNeuronList_.size() > 0)
   {
      // create the data space & dataset for probed neurons
      dims[0] = static_cast<hsize_t>(model->getLayout()->probedNeuronList_.size());
      DataSpace dsProbedNeurons(1, dims);
      dataSetProbedNeurons_ = new DataSet(stateOut_->createDataSet(nameProbedNeurons, PredType::NATIVE_INT, dsProbedNeurons));

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
      cparms.setFillValue( PredType::NATIVE_UINT64, &fill_val);

      // modify dataset creation properties, enable chunking
      hsize_t      chunk_dims[2];
      chunk_dims[0] = static_cast<hsize_t>(100);
      chunk_dims[1] = static_cast<hsize_t>(model->getLayout()->probedNeuronList_.size());
      cparms.setChunk( 2, chunk_dims );

      dataSetSpikesProbedNeurons_ = new DataSet(stateOut_->createDataSet(nameSpikesProbedNeurons, PredType::NATIVE_UINT64, dsSpikesProbedNeurons, cparms));
   }

    // allocate and initialize data memories
    burstinessHist_ = new int[static_cast<int>(Simulator::getInstance().getEpochDuration())];
   spikesHistory_ = new int[static_cast<int>(Simulator::getInstance().getEpochDuration() * 100)];
    memset(burstinessHist_, 0, static_cast<int>(Simulator::getInstance().getEpochDuration() * sizeof(int)));
    memset(spikesHistory_, 0, static_cast<int>(Simulator::getInstance().getEpochDuration() * 100 * sizeof(int)));

    // create the data space & dataset for spikes history of probed neurons
    if (model->getLayout()->probedNeuronList_.size() > 0)
    {
        // allocate data for spikesProbedNeurons
        spikesProbedNeurons_ = new vector<uint64_t>[model->getLayout()->probedNeuronList_.size()];

   // allocate and initialize memory to save offsets of what's been written
   offsetSpikesProbedNeurons_ = new hsize_t[model->getLayout()->probedNeuronList_.size()];
   memset(offsetSpikesProbedNeurons_, 0, static_cast<int>(model->getLayout()->probedNeuronList_.size() * sizeof(hsize_t)));
    }
}

/*
 * Init history matrices with default values
 */
void Hdf5Recorder::initDefaultValues()
{
}

/*
 * Init history matrices with current radii and rates
 */
void Hdf5Recorder::initValues()
{
}

/*
 * Get the current values
 */
void Hdf5Recorder::getValues()
{
}

/*
 * Terminate process
 */
void Hdf5Recorder::term()
{
   // deallocate all objects
   delete[] burstinessHist_;
   delete[] spikesHistory_;

   delete dataSetBurstHist_;
   delete dataSetSpikesHist_;

   shared_ptr<Model> model = Simulator::getInstance().getModel();

   if (model->getLayout()->probedNeuronList_.size() > 0)
   {
      delete dataSetProbedNeurons_;
      delete dataSetSpikesProbedNeurons_;

      delete[] spikesProbedNeurons_;
      delete[] offsetSpikesProbedNeurons_;
   }

   delete stateOut_;
}

/*
 * Compile history information in every epoch.
 *
 * @param[in] neurons   The entire list of neurons.
 */
void Hdf5Recorder::compileHistories(IAllNeurons &neurons)
{
   AllSpikingNeurons &spNeurons = dynamic_cast<AllSpikingNeurons&>(neurons);

   int maxSpikes = (int) ((Simulator::getInstance().getEpochDuration() * Simulator::getInstance().getMaxFiringRate()));

   unsigned int iProbe = 0;    // index into the probedNeuronsLayout vector
   bool fProbe = false;

   shared_ptr<Model> model = Simulator::getInstance().getModel();

   // output spikes: iterate over each neuron
   for (int iNeuron = 0; iNeuron < Simulator::getInstance().getTotalNeurons(); iNeuron++)
   {
      // true if this is a probed neuron
      fProbe = ((iProbe < model->getLayout()->probedNeuronList_.size()) && (iNeuron == model->getLayout()->probedNeuronList_[iProbe]));

      // Point to the current neuron's spike history
      uint64_t* pSpikes = spNeurons.spikeHistory_[iNeuron];

      int& spike_count = spNeurons.spikeCount_[iNeuron];
      int& offset = spNeurons.spikeCountOffset_[iNeuron];
      // iterate over each spike that neuron produced
      for (int i = 0, idxSp = offset; i < spike_count; i++, idxSp++)
      {
         // Single precision (float) gives you 23 bits of significand, 8 bits of exponent, 
         // and 1 sign bit. Double precision (double) gives you 52 bits of significand, 
         // 11 bits of exponent, and 1 sign bit. 
         // Therefore, single precision can only handle 2^23 = 8,388,608 simulation steps 
         // or 8 epochs (1 epoch = 100s, 1 simulation step = 0.1ms).

         if (idxSp >= maxSpikes) idxSp = 0;
         // compile network wide burstiness index data in 1s bins
         int idx1 = static_cast<int>( static_cast<double>( pSpikes[idxSp] ) * Simulator::getInstance().getDeltaT());
         assert(idx1 >= 0 && idx1 < Simulator::getInstance().getEpochDuration());
         burstinessHist_[idx1]++;

         // compile network wide spike count in 10ms bins
         int idx2 = static_cast<int>( static_cast<double>( pSpikes[idxSp] ) * Simulator::getInstance().getDeltaT() * 100);
         spikesHistory_[idx2]++;

         // compile spikes time of the probed neuron (append spikes time)
         if (fProbe)
         {
            spikesProbedNeurons_[iProbe].insert(spikesProbedNeurons_[iProbe].end(), pSpikes[idxSp]);
         }
      }

      if (fProbe)
      {
         iProbe++;
      }
   }

   // clear spike count for all neurons (we've captured their spike information)
   spNeurons.clearSpikeCounts();

   try
   {
      // write burstiness index
      hsize_t offset[2], count[2];
      hsize_t dimsm[2];
      DataSpace* dataspace;
      DataSpace* memspace;

      offset[0] = (Simulator::getInstance().getCurrentStep() - 1) * Simulator::getInstance().getEpochDuration();
      count[0] = Simulator::getInstance().getEpochDuration();
      dimsm[0] = Simulator::getInstance().getEpochDuration();
      memspace = new DataSpace(1, dimsm, NULL);
      dataspace = new DataSpace(dataSetBurstHist_->getSpace());
      dataspace->selectHyperslab(H5S_SELECT_SET, count, offset);
      dataSetBurstHist_->write(burstinessHist_, PredType::NATIVE_INT, *memspace, *dataspace);
      memset(burstinessHist_, 0, static_cast<int>(Simulator::getInstance().getEpochDuration() * sizeof(int)));
      delete dataspace;
      delete memspace;

      // write network wide spike count in 10ms bins
      offset[0] = (Simulator::getInstance().getCurrentStep() - 1) * Simulator::getInstance().getEpochDuration() * 100;
      count[0] = Simulator::getInstance().getEpochDuration() * 100;
      dimsm[0] = Simulator::getInstance().getEpochDuration() * 100;
      memspace = new DataSpace(1, dimsm, NULL);
      dataspace = new DataSpace(dataSetSpikesHist_->getSpace());
      dataspace->selectHyperslab(H5S_SELECT_SET, count, offset);
      dataSetSpikesHist_->write(spikesHistory_, PredType::NATIVE_INT, *memspace, *dataspace);
      memset(spikesHistory_, 0, static_cast<int>(Simulator::getInstance().getEpochDuration() * 100 * sizeof(int)));
      delete dataspace;
      delete memspace;

      // write spikes data of probed neurons
      if (model->getLayout()->probedNeuronList_.size() > 0)
      {
         unsigned int max_size = 0;
         // iterate over each neuron to find the maximum number of spikes for
         // this epoch
         for (unsigned int i = 0; i < model->getLayout()->probedNeuronList_.size(); i++)
         {
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
         for (unsigned int i = 0; i < model->getLayout()->probedNeuronList_.size(); i++)
         {
            dimsm[0] = spikesProbedNeurons_[i].size();
            dimsm[1] = 1;
            memspace = new DataSpace(2, dimsm, NULL);

            offset[0] = offsetSpikesProbedNeurons_[i];
            offset[1] = i;
            count[0] = spikesProbedNeurons_[i].size();
            count[1] = 1;
            dataspace->selectHyperslab(H5S_SELECT_SET, count, offset);
            offsetSpikesProbedNeurons_[i] += spikesProbedNeurons_[i].size();

            dataSetSpikesProbedNeurons_->write(static_cast<uint64_t*>(&(spikesProbedNeurons_[i][0])), PredType::NATIVE_UINT64, *memspace, *dataspace);

            // clear the probed spike data
            spikesProbedNeurons_[i].clear();
            delete memspace;
         }

         delete dataspace;
      }
   }

    // catch failure caused by the H5File operations
    catch( FileIException error )
    {
        error.printErrorStack();
        return;
    }

    // catch failure caused by the DataSet operations
    catch( DataSetIException error )
    {
        error.printErrorStack();
        return;
    }

    // catch failure caused by the DataSpace operations
    catch( DataSpaceIException error )
    {
        error.printErrorStack();
        return;
    }

    // catch failure caused by the DataType operations
    catch( DataTypeIException error )
    {
        error.printErrorStack();
        return;
    }

}

/*
 * Writes simulation results to an output destination.
 *
 * @param  neurons the AllNeurons object.
 **/
void Hdf5Recorder::saveSimData(const IAllNeurons &neurons)
{
   shared_ptr<Model> model = Simulator::getInstance().getModel();

    try
    {
        // create Neuron Types matrix
        VectorMatrix neuronTypes(MATRIX_TYPE, MATRIX_INIT, 1, Simulator::getInstance().getTotalNeurons(), EXC);
        for (int i = 0; i < Simulator::getInstance().getTotalNeurons(); i++) {
            neuronTypes[i] = model->getLayout()->neuronTypeMap_[i];
        }

        // create neuron threshold matrix
        VectorMatrix neuronThresh(MATRIX_TYPE, MATRIX_INIT, 1, Simulator::getInstance().getTotalNeurons(), 0);
        for (int i = 0; i < Simulator::getInstance().getTotalNeurons(); i++) {
            neuronThresh[i] = dynamic_cast<const AllIFNeurons&>(neurons).Vthresh_[i];
        }

        // Write the neuron location matrices
        int* iXloc = new int[Simulator::getInstance().getTotalNeurons()];
        int* iYloc = new int[Simulator::getInstance().getTotalNeurons()];
        for (int i = 0; i < Simulator::getInstance().getTotalNeurons(); i++) {
            // convert VectorMatrix to int array
            iXloc[i] = (*model->getLayout()->xloc_)[i];
            iYloc[i] = (*model->getLayout()->yloc_)[i];
        }
        dataSetXloc_->write(iXloc, PredType::NATIVE_INT);
        dataSetYloc_->write(iYloc, PredType::NATIVE_INT);
        delete[] iXloc;
        delete[] iYloc;

        int* iNeuronTypes = new int[Simulator::getInstance().getTotalNeurons()];
        for (int i = 0; i < Simulator::getInstance().getTotalNeurons(); i++)
        {
            iNeuronTypes[i] = neuronTypes[i];
        }
        dataSetNeuronTypes_->write(iNeuronTypes, PredType::NATIVE_INT);
        delete[] iNeuronTypes;

        int num_starter_neurons = static_cast<int>(model->getLayout()->numEndogenouslyActiveNeurons_);
        if (num_starter_neurons > 0)
        {
            VectorMatrix starterNeurons(MATRIX_TYPE, MATRIX_INIT, 1, num_starter_neurons);
            getStarterNeuronMatrix(starterNeurons, model->getLayout()->starterMap_);

            // create the data space & dataset for starter neurons
            hsize_t dims[2];
            dims[0] = static_cast<hsize_t>(starterNeurons.Size());
            DataSpace dsStarterNeurons(1, dims);
           dataSetStarterNeurons_ = new DataSet(stateOut_->createDataSet(nameStarterNeurons, PredType::NATIVE_INT, dsStarterNeurons));

            int* iStarterNeurons = new int[starterNeurons.Size()];
            for (int i = 0; i < starterNeurons.Size(); i++)
            {
                iStarterNeurons[i] = starterNeurons[i];
            }
            dataSetStarterNeurons_->write(iStarterNeurons, PredType::NATIVE_INT);
            delete[] iStarterNeurons;
            delete dataSetStarterNeurons_;
        }

        // Finalize probed neurons' spikes dataset
        if (model->getLayout()->probedNeuronList_.size() > 0)
        {
           // create the data space & dataset for probed neurons
            hsize_t dims[2];

            int* iProbedNeurons = new int[model->getLayout()->probedNeuronList_.size()];
            for (unsigned int i = 0; i < model->getLayout()->probedNeuronList_.size(); i++)
            {
                iProbedNeurons[i] = model->getLayout()->probedNeuronList_[i];
            }
            dataSetProbedNeurons_->write(iProbedNeurons, PredType::NATIVE_INT);
            delete[] iProbedNeurons;

            // Create the data space for the attribute (unit of the spikes of probed neurons in second).
            dims[0] = 1;
            DataSpace dsAttrPNUnit(1, dims);

            // Create a dataset attribute. 
            Attribute attribute = dataSetSpikesProbedNeurons_->createAttribute(nameAttrPNUnit, H5_FLOAT, dsAttrPNUnit, PropList::DEFAULT);

            // Write the attribute data.
            float deltaT = Simulator::getInstance().getDeltaT();
            attribute.write(H5_FLOAT, &deltaT);
        }

        // Write neuron thresold
        BGFLOAT* fNeuronThresh = new BGFLOAT[Simulator::getInstance().getTotalNeurons()];
        for (int i = 0; i < Simulator::getInstance().getTotalNeurons(); i++)
        {
            fNeuronThresh[i] = neuronThresh[i];
        }
        dataSetNeuronThresh_->write(fNeuronThresh, H5_FLOAT);
        delete[] fNeuronThresh;
    
        // write time between growth cycles
        BGFLOAT epochDuration = Simulator::getInstance().getEpochDuration();
        dataSetTsim_->write(&epochDuration, H5_FLOAT);
        delete dataSetTsim_;

        // write simulation end time
        BGFLOAT endTime = g_simulationStep * Simulator::getInstance().getDeltaT();
        dataSetSimulationEndTime_->write(&endTime, H5_FLOAT);
        delete dataSetSimulationEndTime_;
    }

    // catch failure caused by the DataSet operations
    catch (DataSetIException error)
    {
        error.printErrorStack();
        return;
    }

    // catch failure caused by the DataSpace operations
    catch (DataSpaceIException error)
    {
        error.printErrorStack();
        return;
    }
}

/*
 *  Get starter Neuron matrix.
 *
 *  @param  matrix      Starter Neuron matrix.
 *  @param  starterMap Bool map to reference neuron matrix location from.
 */
void Hdf5Recorder::getStarterNeuronMatrix(VectorMatrix& matrix, const bool* starterMap)
{
    int cur = 0;
    for (int i = 0; i < Simulator::getInstance().getTotalNeurons(); i++) {
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
void Hdf5Recorder::printParameters() {
   LOG4CPLUS_DEBUG(fileLogger_, "\nHDF5 PARAMETERS" << endl
                                                           << "\tResult file path: " << resultFileName_ << endl);
}
