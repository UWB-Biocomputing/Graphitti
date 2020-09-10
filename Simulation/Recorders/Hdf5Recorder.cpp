/*
 *      @file Hdf5Recorder.cpp
 *
 *      @brief An implementation for recording spikes history on hdf5 file
 */
//! An implementation for recording spikes history on hdf5 file

#include "Hdf5Recorder.h"
#include "AllIFNeurons.h"      // TODO: remove LIF model specific code

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
   offsetSpikesProbedNeurons(NULL),
   spikesProbedNeurons(NULL) {
   // TODO: I don't think these member variables exist in the .h file
   resultFileName_ = Simulator::getInstance().getResultFileName();
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
        stateOut = new H5File( resultFileName_, H5F_ACC_TRUNC );

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
   dims[0] = static_cast<hsize_t>(Simulator::getInstance().epochDuration * Simulator::getInstance().maxSteps);
   DataSpace dsBurstHist(1, dims);
   dataSetBurstHist = new DataSet(stateOut->createDataSet(nameBurstHist, PredType::NATIVE_INT, dsBurstHist));

   // create the data space & dataset for spikes history
   dims[0] = static_cast<hsize_t>(Simulator::getInstance().epochDuration * Simulator::getInstance().maxSteps * 100);
   DataSpace dsSpikesHist(1, dims);
   dataSetSpikesHist = new DataSet(stateOut->createDataSet(nameSpikesHist, PredType::NATIVE_INT, dsSpikesHist));

   // create the data space & dataset for xloc & yloc
   dims[0] = static_cast<hsize_t>(Simulator::getInstance().totalNeurons);
   DataSpace dsXYloc(1, dims);
   dataSetXloc = new DataSet(stateOut->createDataSet(nameXloc, PredType::NATIVE_INT, dsXYloc));
   dataSetYloc = new DataSet(stateOut->createDataSet(nameYloc, PredType::NATIVE_INT, dsXYloc));

   // create the data space & dataset for neuron types
   dims[0] = static_cast<hsize_t>(Simulator::getInstance().totalNeurons);
   DataSpace dsNeuronTypes(1, dims);
   dataSetNeuronTypes = new DataSet(stateOut->createDataSet(nameNeuronTypes, PredType::NATIVE_INT, dsNeuronTypes));

   // create the data space & dataset for neuron threshold
   dims[0] = static_cast<hsize_t>(Simulator::getInstance().totalNeurons);
   DataSpace dsNeuronThresh(1, dims);
   dataSetNeuronThresh = new DataSet(stateOut->createDataSet(nameNeuronThresh, H5_FLOAT, dsNeuronThresh));

   // create the data space & dataset for simulation step duration
   dims[0] = static_cast<hsize_t>(1);
   DataSpace dsTsim(1, dims);
   dataSetTsim = new DataSet(stateOut->createDataSet(nameTsim, H5_FLOAT, dsTsim));

   // create the data space & dataset for simulation end time
   dims[0] = static_cast<hsize_t>(1);
   DataSpace dsSimulationEndTime(1, dims);
   dataSetSimulationEndTime = new DataSet(stateOut->createDataSet(nameSimulationEndTime, H5_FLOAT, dsSimulationEndTime));

   // Set up probed neurons so that they can be written incrementally
   if (m_model->getLayout()->m_probed_neuron_list.size() > 0)
   {
      // create the data space & dataset for probed neurons
      dims[0] = static_cast<hsize_t>(m_model->getLayout()->m_probed_neuron_list.size());
      DataSpace dsProbedNeurons(1, dims);
      dataSetProbedNeurons = new DataSet(stateOut->createDataSet(nameProbedNeurons, PredType::NATIVE_INT, dsProbedNeurons));

      // create the data space & dataset for spikes of probed neurons

      // the data space with unlimited dimensions
      hsize_t maxdims[2];
      maxdims[0] = H5S_UNLIMITED;
      maxdims[1] = static_cast<hsize_t>(m_model->getLayout()->m_probed_neuron_list.size());

      // dataset dimensions at creation
      dims[0] = static_cast<hsize_t>(1);
      dims[1] = static_cast<hsize_t>(m_model->getLayout()->m_probed_neuron_list.size());
      DataSpace dsSpikesProbedNeurons(2, dims, maxdims);

      // set fill value for the dataset
      DSetCreatPropList cparms;
      uint64_t fill_val = 0;
      cparms.setFillValue( PredType::NATIVE_UINT64, &fill_val);

      // modify dataset creation properties, enable chunking
      hsize_t      chunk_dims[2];
      chunk_dims[0] = static_cast<hsize_t>(100);
      chunk_dims[1] = static_cast<hsize_t>(m_model->getLayout()->m_probed_neuron_list.size());
      cparms.setChunk( 2, chunk_dims );

      dataSetSpikesProbedNeurons = new DataSet(stateOut->createDataSet(nameSpikesProbedNeurons, PredType::NATIVE_UINT64, dsSpikesProbedNeurons, cparms));
   }

    // allocate and initialize data memories
    burstinessHist = new int[static_cast<int>(Simulator::getInstance().epochDuration)];
    spikesHistory = new int[static_cast<int>(Simulator::getInstance().epochDuration * 100)]; 
    memset(burstinessHist, 0, static_cast<int>(Simulator::getInstance().epochDuration * sizeof(int)));
    memset(spikesHistory, 0, static_cast<int>(Simulator::getInstance().epochDuration * 100 * sizeof(int)));

    // create the data space & dataset for spikes history of probed neurons
    if (m_model->getLayout()->m_probed_neuron_list.size() > 0)
    {
        // allocate data for spikesProbedNeurons
        spikesProbedNeurons = new vector<uint64_t>[m_model->getLayout()->m_probed_neuron_list.size()];

   // allocate and initialize memory to save offsets of what's been written
   offsetSpikesProbedNeurons = new hsize_t[m_model->getLayout()->m_probed_neuron_list.size()];
   memset(offsetSpikesProbedNeurons, 0, static_cast<int>(m_model->getLayout()->m_probed_neuron_list.size() * sizeof(hsize_t)));
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
   delete[] burstinessHist;
   delete[] spikesHistory;

   delete dataSetBurstHist;
   delete dataSetSpikesHist;

   if (m_model->getLayout()->m_probed_neuron_list.size() > 0)
   {
      delete dataSetProbedNeurons;
      delete dataSetSpikesProbedNeurons;

      delete[] spikesProbedNeurons;
      delete[] offsetSpikesProbedNeurons;
   }

   delete stateOut;
}

/*
 * Compile history information in every epoch.
 *
 * @param[in] neurons   The entire list of neurons.
 */
void Hdf5Recorder::compileHistories(IAllNeurons &neurons)
{
   AllSpikingNeurons &spNeurons = dynamic_cast<AllSpikingNeurons&>(neurons);
   int max_spikes = (int) ((Simulator::getInstance().epochDuration * Simulator::getInstance().maxFiringRate));

   unsigned int iProbe = 0;    // index into the probedNeuronsLayout vector
   bool fProbe = false;

   // output spikes: iterate over each neuron
   for (int iNeuron = 0; iNeuron < Simulator::getInstance().totalNeurons; iNeuron++)
   {
      // true if this is a probed neuron
      fProbe = ((iProbe < m_model->getLayout()->m_probed_neuron_list.size()) && (iNeuron == m_model->getLayout()->m_probed_neuron_list[iProbe]));

      // Point to the current neuron's spike history
      uint64_t* pSpikes = spNeurons.spike_history[iNeuron];

      int& spike_count = spNeurons.spikeCount[iNeuron];
      int& offset = spNeurons.spikeCountOffset[iNeuron];
      // iterate over each spike that neuron produced
      for (int i = 0, idxSp = offset; i < spike_count; i++, idxSp++)
      {
         // Single precision (float) gives you 23 bits of significand, 8 bits of exponent, 
         // and 1 sign bit. Double precision (double) gives you 52 bits of significand, 
         // 11 bits of exponent, and 1 sign bit. 
         // Therefore, single precision can only handle 2^23 = 8,388,608 simulation steps 
         // or 8 epochs (1 epoch = 100s, 1 simulation step = 0.1ms).

         if (idxSp >= max_spikes) idxSp = 0;
         // compile network wide burstiness index data in 1s bins
         int idx1 = static_cast<int>( static_cast<double>( pSpikes[idxSp] ) *  Simulator::getInstance().deltaT
               - ( (Simulator::getInstance().currentStep - 1) * Simulator::getInstance().epochDuration ) );
         assert(idx1 >= 0 && idx1 < Simulator::getInstance().epochDuration);
         burstinessHist[idx1]++;

         // compile network wide spike count in 10ms bins
         int idx2 = static_cast<int>( static_cast<double>( pSpikes[idxSp] ) * Simulator::getInstance().deltaT * 100
               - ( (Simulator::getInstance().currentStep - 1) * Simulator::getInstance().epochDuration * 100 ) );
         assert(idx2 >= 0 && idx2 < Simulator::getInstance().epochDuration * 100);
         spikesHistory[idx2]++;

         // compile spikes time of the probed neuron (append spikes time)
         if (fProbe)
         {
            spikesProbedNeurons[iProbe].insert(spikesProbedNeurons[iProbe].end(), pSpikes[idxSp]);
         }
      }

      if (fProbe)
      {
         iProbe++;
      }
   }

   // clear spike count for all neurons (we've captured their spike information)
   spNeurons.clearSpikeCounts(m_sim_info);

   try
   {
      // write burstiness index
      hsize_t offset[2], count[2];
      hsize_t dimsm[2];
      DataSpace* dataspace;
      DataSpace* memspace;

      offset[0] = (Simulator::getInstance().currentStep - 1) * Simulator::getInstance().epochDuration;
      count[0] = Simulator::getInstance().epochDuration;
      dimsm[0] = Simulator::getInstance().epochDuration;
      memspace = new DataSpace(1, dimsm, NULL);
      dataspace = new DataSpace(dataSetBurstHist->getSpace());
      dataspace->selectHyperslab(H5S_SELECT_SET, count, offset);
      dataSetBurstHist->write(burstinessHist, PredType::NATIVE_INT, *memspace, *dataspace); 
      memset(burstinessHist, 0, static_cast<int>(Simulator::getInstance().epochDuration * sizeof(int)));
      delete dataspace;
      delete memspace;

      // write network wide spike count in 10ms bins
      offset[0] = (Simulator::getInstance().currentStep - 1) * Simulator::getInstance().epochDuration * 100;
      count[0] = Simulator::getInstance().epochDuration * 100;
      dimsm[0] = Simulator::getInstance().epochDuration * 100;
      memspace = new DataSpace(1, dimsm, NULL);
      dataspace = new DataSpace(dataSetSpikesHist->getSpace());
      dataspace->selectHyperslab(H5S_SELECT_SET, count, offset);
      dataSetSpikesHist->write(spikesHistory, PredType::NATIVE_INT, *memspace, *dataspace); 
      memset(spikesHistory, 0, static_cast<int>(Simulator::getInstance().epochDuration * 100 * sizeof(int)));
      delete dataspace;
      delete memspace;

      // write spikes data of probed neurons
      if (m_model->getLayout()->m_probed_neuron_list.size() > 0)
      {
         unsigned int max_size = 0;
         // iterate over each neuron to find the maximum number of spikes for
         // this epoch
         for (unsigned int i = 0; i < m_model->getLayout()->m_probed_neuron_list.size(); i++)
         {
            unsigned int size = spikesProbedNeurons[i].size() + offsetSpikesProbedNeurons[i];
            max_size = (max_size > size) ? max_size : size;
         }
         // dataset dimensions
         dimsm[0] = static_cast<hsize_t>(max_size);
         dimsm[1] = static_cast<hsize_t>(m_model->getLayout()->m_probed_neuron_list.size());

         // extend the dataset
         dataSetSpikesProbedNeurons->extend(dimsm);
         dataspace = new DataSpace(dataSetSpikesProbedNeurons->getSpace());

         // write it! Iterate over each neuron's spike data.
         for (unsigned int i = 0; i < m_model->getLayout()->m_probed_neuron_list.size(); i++)
         {
            dimsm[0] = spikesProbedNeurons[i].size();
            dimsm[1] = 1;
            memspace = new DataSpace(2, dimsm, NULL);

            offset[0] = offsetSpikesProbedNeurons[i];
            offset[1] = i;
            count[0] = spikesProbedNeurons[i].size();
            count[1] = 1;
            dataspace->selectHyperslab(H5S_SELECT_SET, count, offset);
            offsetSpikesProbedNeurons[i] += spikesProbedNeurons[i].size(); 

            dataSetSpikesProbedNeurons->write(static_cast<uint64_t*>(&(spikesProbedNeurons[i][0])), PredType::NATIVE_UINT64, *memspace, *dataspace);

            // clear the probed spike data
            spikesProbedNeurons[i].clear();
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
    try
    {
        // create Neuron Types matrix
        VectorMatrix neuronTypes(MATRIX_TYPE, MATRIX_INIT, 1, m_Simulator::getInstance().getTotalNeurons(), EXC);
        for (int i = 0; i < m_Simulator::getInstance().getTotalNeurons(); i++) {
            neuronTypes[i] = m_model->getLayout()->neuron_type_map[i];
        }

        // create neuron threshold matrix
        VectorMatrix neuronThresh(MATRIX_TYPE, MATRIX_INIT, 1, m_Simulator::getInstance().getTotalNeurons(), 0);
        for (int i = 0; i < m_Simulator::getInstance().getTotalNeurons(); i++) {
            neuronThresh[i] = dynamic_cast<const AllIFNeurons&>(neurons).Vthresh[i];
        }

        // Write the neuron location matrices
        int* iXloc = new int[m_Simulator::getInstance().getTotalNeurons()];
        int* iYloc = new int[m_Simulator::getInstance().getTotalNeurons()];
        for (int i = 0; i < m_Simulator::getInstance().getTotalNeurons(); i++) {
            // convert VectorMatrix to int array
            iXloc[i] = (*m_model->getLayout()->xloc)[i];
            iYloc[i] = (*m_model->getLayout()->yloc)[i];
        }
        dataSetXloc->write(iXloc, PredType::NATIVE_INT);
        dataSetYloc->write(iYloc, PredType::NATIVE_INT);
        delete[] iXloc;
        delete[] iYloc;

        int* iNeuronTypes = new int[Simulator::getInstance().totalNeurons];
        for (int i = 0; i < Simulator::getInstance().totalNeurons; i++)
        {
            iNeuronTypes[i] = neuronTypes[i];
        }
        dataSetNeuronTypes->write(iNeuronTypes, PredType::NATIVE_INT);
        delete[] iNeuronTypes;

        int num_starter_neurons = static_cast<int>(m_model->getLayout()->num_endogenously_active_neurons);
        if (num_starter_neurons > 0)
        {
            VectorMatrix starterNeurons(MATRIX_TYPE, MATRIX_INIT, 1, num_starter_neurons);
            getStarterNeuronMatrix(starterNeurons, m_model->getLayout()->starter_map, m_sim_info);

            // create the data space & dataset for starter neurons
            hsize_t dims[2];
            dims[0] = static_cast<hsize_t>(starterNeurons.Size());
            DataSpace dsStarterNeurons(1, dims);
            dataSetStarterNeurons = new DataSet(stateOut->createDataSet(nameStarterNeurons, PredType::NATIVE_INT, dsStarterNeurons));

            int* iStarterNeurons = new int[starterNeurons.Size()];
            for (int i = 0; i < starterNeurons.Size(); i++)
            {
                iStarterNeurons[i] = starterNeurons[i];
            }
            dataSetStarterNeurons->write(iStarterNeurons, PredType::NATIVE_INT);
            delete[] iStarterNeurons;
            delete dataSetStarterNeurons;
        }

        // Finalize probed neurons' spikes dataset
        if (m_model->getLayout()->m_probed_neuron_list.size() > 0)
        {
           // create the data space & dataset for probed neurons
            hsize_t dims[2];

            int* iProbedNeurons = new int[m_model->getLayout()->m_probed_neuron_list.size()];
            for (unsigned int i = 0; i < m_model->getLayout()->m_probed_neuron_list.size(); i++)
            {
                iProbedNeurons[i] = m_model->getLayout()->m_probed_neuron_list[i];
            }
            dataSetProbedNeurons->write(iProbedNeurons, PredType::NATIVE_INT);
            delete[] iProbedNeurons;

            // Create the data space for the attribute (unit of the spikes of probed neurons in second).
            dims[0] = 1;
            DataSpace dsAttrPNUnit(1, dims);

            // Create a dataset attribute. 
            Attribute attribute = dataSetSpikesProbedNeurons->createAttribute(nameAttrPNUnit, H5_FLOAT, dsAttrPNUnit, PropList::DEFAULT);
     
            // Write the attribute data. 
            attribute.write(H5_FLOAT, &(Simulator::getInstance().deltaT));

        }

        // Write neuron thresold
        BGFLOAT* fNeuronThresh = new BGFLOAT[Simulator::getInstance().totalNeurons];
        for (int i = 0; i < Simulator::getInstance().totalNeurons; i++)
        {
            fNeuronThresh[i] = neuronThresh[i];
        }
        dataSetNeuronThresh->write(fNeuronThresh, H5_FLOAT);
        delete[] fNeuronThresh;
    
        // write time between growth cycles
        dataSetTsim->write(&Simulator::getInstance().epochDuration, H5_FLOAT);
        delete dataSetTsim;

        // write simulation end time
        BGFLOAT endtime = g_simulationStep * Simulator::getInstance().deltaT;
        dataSetSimulationEndTime->write(&endtime, H5_FLOAT);
        delete dataSetSimulationEndTime;
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
 *  @param  starter_map Bool map to reference neuron matrix location from.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void Hdf5Recorder::getStarterNeuronMatrix(VectorMatrix& matrix, const bool* starter_map, const SimulationInfo *sim_info)
{
    int cur = 0;
    for (int i = 0; i < sim_info->totalNeurons; i++) {
        if (starter_map[i]) {
            matrix[cur] = i;
            cur++;
        }
    }
}
