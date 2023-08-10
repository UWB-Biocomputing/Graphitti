/**
 * @file Hdf5GrowthRecorder.h
 * 
 * @ingroup Simulator/Recorders
 *
 * @brief An implementation for recording spikes history on hdf5 file
 * 
 * The Hdf5GrowthRecorder provides a mechanism for recording neuron's layout, spikes history,
 * and compile history information on hdf5 file:
 *     -# neuron's locations, and type map,
 *     -# individual neuron's spike rate in epochs,
 *     -# network wide spike count in 10ms bins,
 *     -# individual neuron's radius history of every epoch.
 *
 * Hierarchical Data Format (HDF) is a set of file formats (HDF4, HDF5) designed 
 * to store and organize large amounts of data. 
 */

#pragma once

#if defined(HDF5)

   #include "H5Cpp.h"
   #include "Hdf5Recorder.h"
   #include "Model.h"

   #ifndef H5_NO_NAMESPACE
using namespace H5;
   #endif

   #ifdef SINGLEPRECISION
      #define H5_FLOAT PredType::NATIVE_FLOAT
   #else
      #define H5_FLOAT PredType::NATIVE_DOUBLE
   #endif

   #include <vector>

class Hdf5GrowthRecorder : public Hdf5Recorder {
public:
   /// The constructor and destructor
   Hdf5GrowthRecorder() = default;

   ~Hdf5GrowthRecorder() = default;

   static Recorder *Create()
   {
      return new Hdf5GrowthRecorder();
   }

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


   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() override;

   virtual void registerVariable(std::string varName, EventBuffer &recordVar) override;

protected:
   virtual void initDataSet() override;

   /// Incrementaly write radii and rates histories
   void writeRadiiRates();

   /// hdf5 file dataset
   ///extentable DataSet
   DataSet dataSetRatesHist_;
   DataSet dataSetRadiiHist_;

   /// track radii
   std::vector<BGFLOAT> radiiHistory_;

   /// track firing rate
   std::vector<BGFLOAT> ratesHistory_;
};

#endif   // HDF5
