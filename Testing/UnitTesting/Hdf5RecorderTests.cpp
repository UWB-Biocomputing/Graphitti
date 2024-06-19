#include "Hdf5Recorder.h"
#include "Recorder.h"
#include "gtest/gtest.h"

// Test case for initializing the Hdf5Recorder
TEST(Hdf5RecorderTest, CreateInstanceSuccess)
{
   Recorder *recorder = Hdf5Recorder::Create();
   ASSERT_TRUE(recorder != nullptr);
}

// Test case for open file successfully
TEST(Hdf5RecorderTest, Hdf5InitTest)
{
   // Create an instance of Hdf5Recorder
   std::string outputFile = "../Testing/UnitTesting/TestOutput/Hdf5test_output.h5";
   Hdf5Recorder recorder(outputFile);
   /*recorder.init();
   // Test to see if output file exist
   FILE *f = fopen("../Testing/UnitTesting/TestOutput/Hdf5test_output.h5", "r");
   bool fileExist = f != NULL;
   fclose(f);
   ASSERT_TRUE(fileExist);*/
}