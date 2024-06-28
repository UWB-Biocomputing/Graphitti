/**
 * @file Hdf5Recorder.cpp
 * 
 * @ingroup Simulator/Recorders
 *
 * @brief An implementation for recording spikes history on hdf5 file
 */

#include "Hdf5Recorder.h"
#include "Model.h"
#include "OperationManager.h"
#include "ParameterManager.h"
#include <fstream>
#include <iostream>

#if defined(HDF5)

/// The constructor and destructor
Hdf5Recorder::Hdf5Recorder()
{
   // Retrieve the result file name from the ParameterManager
   ParameterManager::getInstance().getStringByXpath(
      "//RecorderParams/RecorderFiles/resultFileName/text()", resultFileName_);

   // Register the printParameters function with the OperationManager
   function<void()> printParametersFunc = std::bind(&Hdf5Recorder::printParameters, this);
   OperationManager::getInstance().registerOperation(Operations::printParameters,
                                                     printParametersFunc);

   // Initialize the logger for file operations
   fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));

   resultOut_ = nullptr;
}

// destructor
Hdf5Recorder::~Hdf5Recorder()
{
   term();
}

// Other member functions implementation...
void Hdf5Recorder::init()
{
   // Check the output file extension is .h5
   string suffix = ".h5";
   if ((resultFileName_.size() <= suffix.size())
       || (resultFileName_.compare(resultFileName_.size() - suffix.size(), suffix.size(), suffix)
           != 0)) {
      //perror("the file extention is not .h5 ");
      string errorMsg
         = "Error: the file extension is not .h5. Provided file name: " + resultFileName_;
      perror(errorMsg.c_str());
      exit(EXIT_FAILURE);
   }

   // Check if we can create and write to the file
   ofstream testFileWrite(resultFileName_.c_str());
   if (!testFileWrite.is_open()) {
      perror("Error opening output file for writing ");
      exit(EXIT_FAILURE);
   }
   testFileWrite.close();

   try {
      // Create a new file using the default property lists
      resultOut_ = new H5File(resultFileName_, H5F_ACC_TRUNC);
      initDataSet();
   } catch (FileIException &error) {
      cerr << "HDF5 File I/O Exception: " << endl;
      error.printErrorStack();
      return;
   } catch (DataSetIException &error) {
      cerr << "HDF5 Dataset Exception: " << endl;
      error.printErrorStack();
      return;
   } catch (DataSpaceIException &error) {
      cerr << "HDF5 Dataspace Exception: " << endl;
      error.printErrorStack();
      return;
   } catch (DataTypeIException &error) {
      cerr << "HDF5 Datatype Exception: " << endl;
      error.printErrorStack();
      return;
   }
}

// This method closes the HDF5 file and releases any associated resources
void Hdf5Recorder::term()
{
   // checks if the file object `resultOut_` is not null, then attempts to close the file and delete the object
   if (resultOut_ != nullptr) {
      try {
         resultOut_->close();
         delete resultOut_;
         resultOut_ = nullptr;
      } catch (FileIException &error) {
         // If an exception occurs during this process, it prints the error stack for debugging purposes
         cerr << "HDF5 File I/O Exception during termination: ";
         error.printErrorStack();
      }
   }
}

//  Create data spaces and data sets of the hdf5 for recording histories.
void Hdf5Recorder::initDataSet()
{
}
#endif   // HDF5