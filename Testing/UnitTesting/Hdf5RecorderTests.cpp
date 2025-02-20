#include "EventBuffer.h"
#include "Hdf5Recorder.h"
#include "RecordableBase.h"
#include "RecordableVector.h"
#include "Recorder.h"
#include "VectorMatrix.h"
#include "gtest/gtest.h"

#if defined(HDF5)
   #include "H5Cpp.h"

// Test case for initializing the Hdf5Recorder
TEST(Hdf5RecorderTest, CreateInstanceSuccess)
{
   Recorder *recorder = Hdf5Recorder::Create();
   ASSERT_TRUE(recorder != nullptr);
}

// Test case for init() and term()
TEST(Hdf5RecorderTest, Hdf5InitAndTermTest)
{
   // Create an instance of Hdf5Recorder with a specific output file name
   std::string outputFile = "../Testing/UnitTesting/TestOutput/Hdf5test_output_term.h5";
   Hdf5Recorder recorder(outputFile);
   recorder.init();

   // Ensure the file has been created successfully by the constructor
   FILE *f = fopen(outputFile.c_str(), "r");
   ASSERT_TRUE(f != NULL);
   fclose(f);

   // Call the term() method to close the HDF5 file
   recorder.term();

   // Check if the file can be reopened, indicating it was closed properly
   f = fopen(outputFile.c_str(), "r");
   bool fileExist = f != NULL;
   fclose(f);
   ASSERT_TRUE(fileExist);
}

// Unit test for verifying the registerVariable method
TEST(Hdf5RecorderTest, RegisterVariableTest)
{
   // Create an instance of Hdf5Recorder
   std::string outputFile = "../Testing/UnitTesting/TestOutput/Hdf5test_output_register.h5";
   Hdf5Recorder recorder(outputFile);
   recorder.init();

   // Create an EventBuffer for testing
   EventBuffer eventBuffer;
   const H5std_string hdf5Name("test_var");

   // Register the variable
   recorder.registerVariable(hdf5Name, eventBuffer, Recorder::UpdatedType::CONSTANT);

   // Retrieve the registered variable info
   const Hdf5Recorder::hdf5VariableInfo &varInfo = recorder.getVariableTable()[0];

   // Verify the variable type
   ASSERT_EQ(Recorder::UpdatedType::CONSTANT, varInfo.variableType_);
   // Verify the variable name
   ASSERT_EQ(hdf5Name, varInfo.variableName_);
   // Verify the HDF5 data type
   ASSERT_EQ(PredType::NATIVE_UINT64, varInfo.hdf5Datatype_);
}

// Unit test for registerVariable method with a vector of RecordableBase
TEST(Hdf5RecorderTest, RegisterVectorVariableTest)
{
   // Create an instance of Hdf5Recorder
   std::string outputFile = "../Testing/UnitTesting/TestOutput/Hdf5test_output_register.h5";
   Hdf5Recorder recorder(outputFile);
   recorder.init();

   // Create mock EventBuffer objects for testing
   EventBuffer buffer0;
   EventBuffer buffer1;

   // Create a vector of pointers to EventBuffer objects
   std::vector<RecordableBase *> bufferPointers = {&buffer0, &buffer1};

   // Register variables
   recorder.registerVariable("neuron_", bufferPointers, Recorder::UpdatedType::DYNAMIC);

   // Verify that the registered variables are stored correctly
   const auto &variableTable = recorder.getVariableTable();
   ASSERT_EQ(2, variableTable.size());

   ASSERT_EQ("neuron_0", variableTable[0].variableName_);
   ASSERT_EQ("neuron_1", variableTable[1].variableName_);
   ASSERT_EQ(&buffer0, &variableTable[0].variableLocation_);
   ASSERT_EQ(&buffer1, &variableTable[1].variableLocation_);
   ASSERT_EQ(Recorder::UpdatedType::DYNAMIC, variableTable[0].variableType_);
   ASSERT_EQ(Recorder::UpdatedType::DYNAMIC, variableTable[1].variableType_);
}

// Unit test for registerVariable method with a vector of NeuronType enums
TEST(Hdf5RecorderTest, RegisterVertexTypeTest)
{
   // Create an instance of Hdf5Recorder
   std::string outputFile = "../Testing/UnitTesting/TestOutput/Hdf5test_output_register.h5";
   Hdf5Recorder recorder(outputFile);
   recorder.init();

   // Create a vector of NeuronType enums
   RecordableVector<vertexType> neuronTypes;
   neuronTypes.resize(2);
   neuronTypes[0] = vertexType::EXC;  
   neuronTypes[1] = vertexType::INH;  

   // register the vector of NeuronTypes
   recorder.registerVariable("neuron_types", neuronTypes, Recorder::UpdatedType::DYNAMIC);

   // Verify that the registered variables are stored correctly
   const auto &variableTable = recorder.getVariableTable();
   ASSERT_EQ(1, variableTable.size());  // Only one variable, "neuron_types"

   // Verify that the registered variable name matches
   ASSERT_EQ("neuron_types", variableTable[0].variableName_);
   ASSERT_EQ(&neuronTypes, &variableTable[0].variableLocation_);

   // Verify the type of update for this variable
   ASSERT_EQ(Recorder::UpdatedType::DYNAMIC, variableTable[0].variableType_);
}


// Define the test case for saving simulation data
TEST(Hdf5RecorderTest, SaveSimDataTest)
{
   // Define a temporary file path for testing
   std::string outputFile = "../Testing/UnitTesting/TestOutput/Hdf5test_output_save.h5";

   // Create an instance of Hdf5Recorder
   Hdf5Recorder recorder(outputFile);
   recorder.init();

   // Create and configure EventBuffer for testing
   EventBuffer eventBuffer(5);   // Initialize with a size that matches the mock data
   eventBuffer.insertEvent(1);
   eventBuffer.insertEvent(2);
   eventBuffer.insertEvent(3);
   eventBuffer.insertEvent(4);
   eventBuffer.insertEvent(5);

   // Register the variable with Hdf5Recorder
   recorder.registerVariable("test_var1", eventBuffer, Recorder::UpdatedType::CONSTANT);

   // Call saveSimData() to write the data to the file
   recorder.saveSimData();

   // Open the HDF5 file and read back the data
   H5File file(outputFile, H5F_ACC_RDONLY);
   DataSet dataset = file.openDataSet("test_var1");
   DataSpace dataspace = dataset.getSpace();

   hsize_t num_elements;
   dataspace.getSimpleExtentDims(&num_elements, nullptr);

   vector<uint64_t> dataBuffer(num_elements);
   // Read the data into the buffer
   dataset.read(dataBuffer.data(), PredType::NATIVE_UINT64);

   // Verify the data matches the expected values
   vector<uint64_t> expectedData = {1, 2, 3, 4, 5};
   ASSERT_EQ(expectedData.size(), dataBuffer.size());
   for (size_t i = 0; i < expectedData.size(); ++i) {
      EXPECT_EQ(expectedData[i], dataBuffer[i]);
   }
}

// Unit test for saving simulation data with a vector of NeuronType enums
TEST(Hdf5RecorderTest, SaveSimDataVertexTypeTest)
{
   // Define a temporary file path for testing
   std::string outputFile = "../Testing/UnitTesting/TestOutput/Hdf5test_output_save.h5";

   // Create an instance of Hdf5Recorder
   Hdf5Recorder recorder(outputFile);
   recorder.init();

   // Create and configure RecordableVector<vertexType> for testing
   RecordableVector<vertexType> neuronTypes;
   neuronTypes.resize(3);
   neuronTypes[0] = vertexType::EXC;  
   neuronTypes[1] = vertexType::INH; 
   neuronTypes[2] = vertexType::EXC;  

   // Register the variable with Hdf5Recorder
   recorder.registerVariable("neuron_types", neuronTypes, Recorder::UpdatedType::CONSTANT);

   // Call saveSimData() to write the data to the file
   recorder.saveSimData();

   // Open the HDF5 file and read back the data
   H5File file(outputFile, H5F_ACC_RDONLY);
   DataSet dataset = file.openDataSet("neuron_types");
   DataSpace dataspace = dataset.getSpace();

   hsize_t num_elements;
   dataspace.getSimpleExtentDims(&num_elements, nullptr);

   // Read the data into a buffer
   vector<int> dataBuffer(num_elements);
   dataset.read(dataBuffer.data(), PredType::NATIVE_INT);

   // Verify the data matches the expected NeuronType values (converted to int)
   vector<int> expectedData = {static_cast<int>(vertexType::EXC), 
                               static_cast<int>(vertexType::INH), 
                               static_cast<int>(vertexType::EXC)};
   
   ASSERT_EQ(expectedData.size(), dataBuffer.size());
   for (size_t i = 0; i < expectedData.size(); ++i) {
      EXPECT_EQ(expectedData[i], dataBuffer[i]);
   }
}

// Define the test case for compiling histories
TEST(Hdf5RecorderTest, CompileHistoriesTest)
{
   // Define a temporary file path for testing
   std::string outputFile
      = "../Testing/UnitTesting/TestOutput/Hdf5test_output_compile_histories.h5";

   // Create an instance of Hdf5Recorder
   Hdf5Recorder recorder(outputFile);
   recorder.init();

   // Create and configure variables for testing
   EventBuffer eventBufferInt(5);   // Example with int type

   // Register the variable with Hdf5Recorder as DYNAMIC
   recorder.registerVariable("test_var_int", eventBufferInt, Recorder::UpdatedType::DYNAMIC);


   // Call compileHistories() multiple times to simulate multiple epochs
   for (int epoch = 0; epoch < 3; ++epoch) {
      // Clear and insert new events to simulate new data each epoch
      eventBufferInt.clear();
      eventBufferInt.insertEvent(1 * (epoch + 1));
      eventBufferInt.insertEvent(2 * (epoch + 1));
      eventBufferInt.insertEvent(3 * (epoch + 1));
      eventBufferInt.insertEvent(4 * (epoch + 1));
      eventBufferInt.insertEvent(5 * (epoch + 1));

      recorder.compileHistories();
   }

   // Open the HDF5 file and read back the data
   H5File file(outputFile, H5F_ACC_RDONLY);
   DataSet dataset = file.openDataSet("test_var_int");
   DataSpace dataspace = dataset.getSpace();

   hsize_t num_elements;
   dataspace.getSimpleExtentDims(&num_elements, nullptr);

   std::vector<int> dataBuffer(num_elements);
   // Read the data into the buffer
   dataset.read(dataBuffer.data(), PredType::NATIVE_INT);

   // Verify the data matches the expected values (repeated 3 times)
   std::vector<int> expectedData = {1, 2, 3, 4, 5, 2, 4, 6, 8, 10, 3, 6, 9, 12, 15};
   ASSERT_EQ(expectedData.size(), dataBuffer.size());
   for (size_t i = 0; i < expectedData.size(); ++i) {
      EXPECT_EQ(expectedData[i], dataBuffer[i]);
   }
}

// Define the test case for compiling histories with vertexType enum
TEST(Hdf5RecorderTest, CompileHistoriesVertexTypeTest)
{
   // Define temporary file path for testing
   std::string outputFile
      = "../Testing/UnitTesting/TestOutput/Hdf5test_output_compile_histories_neuron_type.h5";

   // Create an instance of Hdf5Recorder
   Hdf5Recorder recorder(outputFile);
   recorder.init();

   // Create and configure EventBuffer for testing (stored as int)
   EventBuffer eventBufferNeuron(5);

   // Register the variable with Hdf5Recorder as DYNAMIC
   recorder.registerVariable("neuron_types", eventBufferNeuron, Recorder::UpdatedType::DYNAMIC);

   // Expected values for checking correctness
   std::vector<int> expectedData;

   // Call compileHistories() multiple times to simulate multiple epochs
   for (int epoch = 0; epoch < 3; ++epoch) {
      // Clear and insert new NeuronType values 
      eventBufferNeuron.clear();
      eventBufferNeuron.insertEvent(static_cast<int>(vertexType::EXC));
      eventBufferNeuron.insertEvent(static_cast<int>(vertexType::INH));
      eventBufferNeuron.insertEvent(static_cast<int>(vertexType::EXC));
      eventBufferNeuron.insertEvent(static_cast<int>(vertexType::EXC));
      eventBufferNeuron.insertEvent(static_cast<int>(vertexType::INH));

      // Append expected values for this epoch
      expectedData.push_back(static_cast<int>(vertexType::EXC));
      expectedData.push_back(static_cast<int>(vertexType::INH));
      expectedData.push_back(static_cast<int>(vertexType::EXC));
      expectedData.push_back(static_cast<int>(vertexType::EXC));
      expectedData.push_back(static_cast<int>(vertexType::INH));

      // Call compile history
      recorder.compileHistories();
   }

   // Open the HDF5 file and read back the data
   H5File file(outputFile, H5F_ACC_RDONLY);
   DataSet dataset = file.openDataSet("neuron_types");
   DataSpace dataspace = dataset.getSpace();

   hsize_t num_elements;
   dataspace.getSimpleExtentDims(&num_elements, nullptr);

   std::vector<int> dataBuffer(num_elements);
   dataset.read(dataBuffer.data(), PredType::NATIVE_INT);

   // Ensure data size matches expectation
   ASSERT_EQ(expectedData.size(), dataBuffer.size());

   // Verify that stored values match expected values
   for (size_t i = 0; i < expectedData.size(); ++i) {
      EXPECT_EQ(expectedData[i], dataBuffer[i]);
   }
}
#endif   // HDF5