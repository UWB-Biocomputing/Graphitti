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
   // I just comment this for now because something wrong with my init function
   //init();
}

// Other member functions implementation...
void Hdf5Recorder::init()
{
   // Check the output file extension is .h5
   /*string suffix = ".h5";
    if ((resultFileName_.size() <= suffix.size()) ||
        (resultFileName_.compare(resultFileName_.size() - suffix.size(), suffix.size(), suffix) != 0)) {
        std::cerr << "The file extension is not .h5" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Check if we can create and write to the file
    std::ofstream testFileWrite(resultFileName_);
    if (!testFileWrite.is_open()) {
        std::cerr << "Error opening output file for writing" << std::endl;
        exit(EXIT_FAILURE);
    }
    testFileWrite.close();

    try {
        // Create a new file using the default property lists
        resultOut_ = H5File(resultFileName_, H5F_ACC_TRUNC);
        initDataSet();
    } catch (const FileIException& error) {
        std::cerr << "HDF5 File I/O Exception: ";
        error.printErrorStack();
        return;
    } catch (const DataSetIException& error) {
        std::cerr << "HDF5 Dataset Exception: ";
        error.printErrorStack();
        return;
    } catch (const DataSpaceIException& error) {
        std::cerr << "HDF5 Dataspace Exception: ";
        error.printErrorStack();
        return;
    } catch (const DataTypeIException& error) {
        std::cerr << "HDF5 Datatype Exception: ";
        error.printErrorStack();
        return;
    }*/
}
#endif   // HDF5