/**
 * @file Hdf5GrowthRecorder.cpp
 * 
 * @ingroup Simulator/Recorders
 *
 * @brief An implementation for recording spikes history on hdf5 file
 */

#if defined(HDF5)

   #include "Hdf5GrowthRecorder.h"
   #include "AllIFNeurons.h"   // TODO: remove LIF model specific code
   #include "ConnGrowth.h"

// hdf5 dataset name
const H5std_string nameRatesHist("ratesHistory");
const H5std_string nameRadiiHist("radiiHistory");

///  Create data spaces and data sets of the hdf5 for recording histories.
void Hdf5GrowthRecorder::initDataSet()
{
   Hdf5Recorder::initDataSet();
   Simulator &simulator = Simulator::getInstance();

   // create the data space & dataset for rates history
   hsize_t dims[2];
   dims[0] = static_cast<hsize_t>(simulator.getNumEpochs() + 1);
   dims[1] = static_cast<hsize_t>(simulator.getTotalVertices());
   DataSpace dsRatesHist(2, dims);
   dataSetRatesHist_ = new DataSet(resultOut_->createDataSet(nameRatesHist, H5_FLOAT, dsRatesHist));

   // create the data space & dataset for radii history
   dims[0] = static_cast<hsize_t>(simulator.getNumEpochs() + 1);
   dims[1] = static_cast<hsize_t>(simulator.getTotalVertices());
   DataSpace dsRadiiHist(2, dims);
   dataSetRadiiHist_ = new DataSet(resultOut_->createDataSet(nameRadiiHist, H5_FLOAT, dsRadiiHist));

   // allocate data memories
   ratesHistory_ = new BGFLOAT[simulator.getTotalVertices()];
   radiiHistory_ = new BGFLOAT[simulator.getTotalVertices()];
}

/// Init radii and rates history matrices with default values
void Hdf5GrowthRecorder::initDefaultValues()
{
   Simulator &simulator = Simulator::getInstance();
   Model *model = simulator.getModel();

   Connections *connections = model->getConnections();
   BGFLOAT startRadius = dynamic_cast<ConnGrowth *>(connections)->growthParams_.startRadius;

   for (int i = 0; i < simulator.getTotalVertices(); i++) {
      radiiHistory_[i] = startRadius;
      ratesHistory_[i] = 0;
   }

   // write initial radii and rate
   // because compileHistories function is not called when simulation starts
   writeRadiiRates();
}

/// Init radii and rates history matrices with current radii and rates
void Hdf5GrowthRecorder::initValues()
{
   Simulator &simulator = Simulator::getInstance();
   Model *model = simulator.getModel();

   Connections *connections = model->getConnections();

   for (int i = 0; i < simulator.getTotalVertices(); i++) {
      radiiHistory_[i] = (dynamic_cast<ConnGrowth *>(connections)->radii_)[i];
      ratesHistory_[i] = (dynamic_cast<ConnGrowth *>(connections)->rates_)[i];
   }

   // write initial radii and rate
   // because compileHistories function is not called when simulation starts
   writeRadiiRates();
}

/// Get the current radii and rates values
void Hdf5GrowthRecorder::getValues()
{
   Model *model = Simulator::getInstance().getModel();
   Connections *connections = model->getConnections();

   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      (dynamic_cast<ConnGrowth *>(connections)->radii_)[i] = radiiHistory_[i];
      (dynamic_cast<ConnGrowth *>(connections)->rates_)[i] = ratesHistory_[i];
   }
}

/// Terminate process
void Hdf5GrowthRecorder::term()
{
   // deallocate all objects
   delete[] ratesHistory_;
   delete[] radiiHistory_;

   Hdf5Recorder::term();
}

/// Compile history information in every epoch.
///
/// @param[in] neurons   The entire list of neurons.
void Hdf5GrowthRecorder::compileHistories(AllVertices &neurons)
{
   Hdf5Recorder::compileHistories(neurons);

   Model *model = Simulator::getInstance().getModel();
   Connections *connections = model->getConnections();

   BGFLOAT minRadius = dynamic_cast<ConnGrowth *>(connections)->growthParams_.minRadius;
   VectorMatrix &rates = (dynamic_cast<ConnGrowth *>(connections)->rates_);
   VectorMatrix &radii = (dynamic_cast<ConnGrowth *>(connections)->radii_);

   // output spikes
   for (int iVertex = 0; iVertex < Simulator::getInstance().getTotalVertices(); iVertex++) {
      // record firing rate to history matrix
      ratesHistory_[iVertex] = rates[iVertex];

      // Cap minimum radius size and record radii to history matrix
      // TODO: find out why we cap this here.
      if (radii[iVertex] < minRadius)
         radii[iVertex] = minRadius;

      // record radius to history matrix
      radiiHistory_[iVertex] = radii[iVertex];

      // ToDo: change this to
      DEBUG_MID(cout << "radii[" << iVertex << ":" << radii[iVertex] << "]" << endl;)
   }

   writeRadiiRates();
}

/// Incrementaly write radii and rates histories
void Hdf5GrowthRecorder::writeRadiiRates()
{
   try {
      // Write radii and rates histories information:
      hsize_t offset[2], count[2];
      hsize_t dimsm[2];
      DataSpace *dataspace;
      DataSpace *memspace;

      // write radii history
      offset[0] = Simulator::getInstance().getCurrentStep();
      offset[1] = 0;
      count[0] = 1;
      count[1] = Simulator::getInstance().getTotalVertices();
      dimsm[0] = 1;
      dimsm[1] = Simulator::getInstance().getTotalVertices();
      memspace = new DataSpace(2, dimsm, nullptr);
      dataspace = new DataSpace(dataSetRadiiHist_->getSpace());
      dataspace->selectHyperslab(H5S_SELECT_SET, count, offset);
      dataSetRadiiHist_->write(radiiHistory_, H5_FLOAT, *memspace, *dataspace);
      delete dataspace;
      delete memspace;

      // write rates history
      offset[0] = Simulator::getInstance().getCurrentStep();
      offset[1] = 0;
      count[0] = 1;
      count[1] = Simulator::getInstance().getTotalVertices();
      dimsm[0] = 1;
      dimsm[1] = Simulator::getInstance().getTotalVertices();
      memspace = new DataSpace(2, dimsm, nullptr);
      dataspace = new DataSpace(dataSetRadiiHist_->getSpace());
      dataspace->selectHyperslab(H5S_SELECT_SET, count, offset);
      dataSetRatesHist_->write(ratesHistory_, H5_FLOAT, *memspace, *dataspace);
      delete dataspace;
      delete memspace;
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

///  Prints out all parameters to logging file.
///  Registered to OperationManager as Operation::printParameters
void Hdf5GrowthRecorder::printParameters()
{
   LOG4CPLUS_DEBUG(fileLogger_, "\n---Hdf5GrowthRecorder Parameters---"
                                   << endl
                                   << "\tRecorder type: Hdf5GrowthRecorder" << endl);
}


#endif   // HDF5
