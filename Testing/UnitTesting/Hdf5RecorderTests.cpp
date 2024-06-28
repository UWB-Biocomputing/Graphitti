#include "Hdf5Recorder.h"
#include "Recorder.h"
#include "gtest/gtest.h"

#if defined(HDF5)

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