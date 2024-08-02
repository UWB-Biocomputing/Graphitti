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
   const Hdf5Recorder::singleVariableInfo &varInfo = recorder.getVariableTable()[0];

   // Verify the variable type
   ASSERT_EQ(Recorder::UpdatedType::CONSTANT, varInfo.variableType_);
   // Verify the variable name
   ASSERT_EQ(hdf5Name, varInfo.variableName_);
   // Verify the HDF5 data type
   ASSERT_EQ(PredType::NATIVE_UINT64, varInfo.hdf5Datatype_);
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

   // Create a unique_ptr to an empty AllVertices object
   unique_ptr<AllVertices> vertices = 0;

   // Call saveSimData() to write the data to the file
   recorder.saveSimData(*vertices);

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

   // Create a unique_ptr to an empty AllVertices object
   unique_ptr<AllVertices> vertices = 0;

   // Call compileHistories() multiple times to simulate multiple epochs
   for (int epoch = 0; epoch < 3; ++epoch) {
      // Clear and insert new events to simulate new data each epoch
      eventBufferInt.clear();
      eventBufferInt.insertEvent(1 * (epoch + 1));
      eventBufferInt.insertEvent(2 * (epoch + 1));
      eventBufferInt.insertEvent(3 * (epoch + 1));
      eventBufferInt.insertEvent(4 * (epoch + 1));
      eventBufferInt.insertEvent(5 * (epoch + 1));

      recorder.compileHistories(*vertices);
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

#endif   // HDF5