#include "EventBuffer.h"
#include "Hdf5Recorder.h"
#include "RecordableBase.h"
#include "RecordableVector.h"
#include "Recorder.h"
#include "gtest/gtest.h"

#if defined(HDF5)
   #include "H5Cpp.h"

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
#endif   // HDF5