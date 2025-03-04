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
   virtual ~Hdf5Recorder();

   /**
   * @brief Factory method to create a new HDF5Recorder instance.
   *
   * @return A pointer to a new HDF5Recorder instance.
   */
   static Recorder *Create()
   {
      return new Hdf5Recorder();
   }

   // Other member functions...
   /// Initialize data
   /// @param[in] stateOutputFileName File name to save histories
   virtual void init() override;

   /// Terminate process
   virtual void term() override;


   /// Compile/capture variable history information in every epoch
   virtual void compileHistories() override;


   /// Writes simulation results to an output destination.
   virtual void saveSimData() override;

   /// Prints out all parameters to logging file.
   /// Registered to OperationManager as Operation::printParameters
   virtual void printParameters() override
   {
   }

   /// Receives a recorded variable entity from the variable owner class
   /// used when the return type from recordable variable is supported by Recorder
   /**
   * @brief Registers a single instance of a class derived from RecordableBase.
   * @param varName Name of the recorded variable.
   * @param recordVar Reference to the recorded variable.
   * @param variableType Type of the recorded variable.
   */
   virtual void registerVariable(const string &varName, RecordableBase &recordVar,
                                 UpdatedType variableType) override;

   /// Register a vector of instance of a class derived from RecordableBase.
   virtual void registerVariable(const string &varName, vector<RecordableBase *> &recordVars,
                                 UpdatedType variableType) override;


   struct hdf5VariableInfo {
      /// the name of each variable
      string variableName_;

      /// the basic data type in the Recorded variable
      string dataType_;

      /// the data type in Hdf5 file
      DataType hdf5Datatype_;

      /// the data set for Hdf5 file
      DataSet hdf5DataSet_;

      /// the variable type: updated frequency
      UpdatedType variableType_;

      /// a reference to a RecordableBase variable
      /// As the simulator runs, the values in the RecordableBase object will be updated
      RecordableBase &variableLocation_;

      // Constructor accepting the variable name, the address of recorded variable, the updated type
      hdf5VariableInfo(const string &name, RecordableBase &location, UpdatedType variableType) :
         variableLocation_(location), variableName_(name), variableType_(variableType)
      {
         dataType_ = location.getDataType();
         convertType();
      }

      // Converts the data type of the variable to HDF5 data type
      void convertType()
      {
         if (dataType_ == typeid(uint64_t).name()) {
            hdf5Datatype_ = PredType::NATIVE_UINT64;
         } else if (dataType_ == typeid(bool).name()) {
            hdf5Datatype_ = PredType::NATIVE_INT;
         } else if (dataType_ == typeid(int).name()) {
            hdf5Datatype_ = PredType::NATIVE_INT;
         } else if (dataType_ == typeid(float).name()) {
            hdf5Datatype_ = PredType::NATIVE_FLOAT;
         } else if (dataType_ == typeid(double).name()) {
            hdf5Datatype_ = PredType::NATIVE_DOUBLE;
         } else if (dataType_ == typeid(vertexType).name()) {
            hdf5Datatype_ = PredType::NATIVE_INT;
         } else {
            throw runtime_error("Unsupported data type");
         }
      }

      // Method to capture and write data to the HDF5 dataset
      void captureData()
      {
         // Ensure the dataset exists and data can be written
         if (variableLocation_.getNumElements() > 0) {
            // Prepare the data buffer based on the HDF5 data type
            if (hdf5Datatype_ == PredType::NATIVE_FLOAT) {
               vector<float> dataBuffer(variableLocation_.getNumElements());
               for (int i = 0; i < variableLocation_.getNumElements(); ++i) {
                  dataBuffer[i] = get<float>(variableLocation_.getElement(i));
               }
               // Write the data to the dataset
               hdf5DataSet_.write(dataBuffer.data(), hdf5Datatype_);

            } else if (hdf5Datatype_ == PredType::NATIVE_INT) {
               vector<int> dataBuffer(variableLocation_.getNumElements());
               for (int i = 0; i < variableLocation_.getNumElements(); ++i) {
                  // For type int, a distinction needs to be made between vertexType and regular integers
                  // Since vertexType is an enum class, it needs to first be converted to an integer before storing
                     
                  // 'decltype' determines the type at compile time 
                  // 'decay_t' removes any references/const from the type
                  // This simplifies comparisons through 'is_same_v' to ensure type matches vertexType
                  if (std::is_same_v<vertexType, std::decay_t<decltype(get<vertexType>(
                                                              variableLocation_.getElement(i)))>>) {
                     // If type matches vertexType, convert to int before storing
                     dataBuffer[i]
                        = static_cast<int>(get<vertexType>(variableLocation_.getElement(i)));
                  } else {
                     // Otherwise, store as a regular integer 
                     dataBuffer[i] = get<int>(variableLocation_.getElement(i));
                  }
               }
               hdf5DataSet_.write(dataBuffer.data(), hdf5Datatype_);

            } else if (hdf5Datatype_ == PredType::NATIVE_UINT64) {
               vector<uint64_t> dataBuffer(variableLocation_.getNumElements());
               for (int i = 0; i < variableLocation_.getNumElements(); ++i) {
                  dataBuffer[i] = get<uint64_t>(variableLocation_.getElement(i));
               }
               hdf5DataSet_.write(dataBuffer.data(), hdf5Datatype_);

            } else {
               // Throw an error if the data type is unsupported
               throw runtime_error("Unsupported data type");
            }
         }
      }
   };

   ///@{
   /** These methods are intended only for unit tests */
   /// constructor only for unit test
   Hdf5Recorder(const string &fileName)
   {
      resultFileName_ = fileName;
   }

   // Accessor method to retrieve variableTable_
   const vector<hdf5VariableInfo> &getVariableTable() const
   {
      return variableTable_;
   }

private:
   /// Populates Starter neuron matrix based with boolean values based on starterMap state
   ///@param[in] matrix  starter neuron matrix
   ///@param starterMap  Bool vector to reference neuron matrix location from.
   virtual void getStarterNeuronMatrix(VectorMatrix &matrix,
                                       const vector<bool> &starterMap) override
   {
   }

   // Member variables for HDF5 datasets
   H5File *resultOut_;

   /// List of registered variables for recording
   vector<hdf5VariableInfo> variableTable_;
   // Other member functions...
};

#endif   // HDF5