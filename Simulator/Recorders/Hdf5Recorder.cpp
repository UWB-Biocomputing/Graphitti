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

   // Initialize the HDF5 file object to nullptr
   // This is the HDF5 file (H5File) object.
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

// create the dataset for constant variable and store the data to dataset
void Hdf5Recorder::saveSimData(const AllVertices &neurons)
{
   // Initialize datasets for constant variables
   for (auto &variableInfo : variableTable_) {
      if (variableInfo.variableType_ == UpdatedType::CONSTANT) {
         // Define dimensions for the constant dataset
         hsize_t constantDims[1]
            = {static_cast<hsize_t>(variableInfo.variableLocation_.getNumElements())};
         DataSpace constantSpace(1, constantDims);

         // Create dataset
         variableInfo.hdf5DataSet_ = resultOut_->createDataSet(
            variableInfo.variableName_, variableInfo.hdf5Datatype_, constantSpace);
         variableInfo.captureData();
      }
   }
}

/// Receives a recorded variable entity from the variable owner class
/// used when the return type from recordable variable is supported by Recorder
/**
* @brief Registers a single instance of a class derived from RecordableBase.
* @param varName Name of the recorded variable.
* @param recordVar Reference to the recorded variable.
* @param variableType Type of the recorded variable.
*/
void Hdf5Recorder::registerVariable(const string &varName, RecordableBase &recordVar,
                                    UpdatedType variableType)
{
   // Create a singleVariableInfo object for the variable
   singleVariableInfo hdf5VarInfo(varName, recordVar, variableType);

   // Add the variable information to the variableTable_
   variableTable_.push_back(hdf5VarInfo);
}
#endif   // HDF5