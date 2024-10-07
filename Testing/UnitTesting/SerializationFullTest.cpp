/**
 * @file SerializationFullTest.cpp
 *
 * @brief The serialization test verifies correctness through a series of steps performed across three files:
 *    SerializeFullFileTest.cpp, SerializationFirstHalfTest.cpp, and SerializationSecondHalfTest.cpp.
 *    
 *    STEP 1: Run a full simulation (e.g., 10 epochs) and save its serialized output.
 *    STEP 2: Split the full simulation's input configuration file into two halves based on the epoch count 
 *            (e.g., 10 epochs → 5 + 5 epochs).
 *            Reference file: /configfiles/test-small-long-half.xml
 *    STEP 3: Run the first half of the simulation (e.g., 5 epochs) and save its serialized output.
 *    STEP 4: Run the second half of the simulation (e.g., the remaining 5 epochs) using the serialized output 
 *            from the first half as the starting point.
 *    STEP 5: Compare the serialized output of the second half with the full simulation's serialized output.
 *            If they match, the final result files should also be identical, confirming successful serialization.
 * 
 * @note This specific test, SerializeFullFileTest, covers the first step: running the full simulation and saving the serialized file.
 * 
 * To run the serialization tests, execute the script `run_serial_test.sh` located in the `build` directory.
 * 
 * @ingroup Testing/UnitTesting
 */

#include "SerializationHelper.cpp"
#include "gtest/gtest.h"

using namespace std;

// Test to ensure the full simulation runs, serializes correctly, and the serialized file is created.
TEST(SerializationFull, SerializeFullFileTest)
{
   string executable = "./cgraphitti";

   // Configuration file for the full simulation
   string configFileFull = "../configfiles/test-small-long.xml";

   // Path to save the serialized output file
   string serialFull = "../Testing/UnitTesting/TestOutput/Full-serialized-file.xml";

   // Command-line arguments for the simulation
   string argumentFull = "-c " + configFileFull + " -s " + serialFull;

   // Run the full simulation
   ASSERT_TRUE(runSimulation(executable, argumentFull)) << "Full simulation failed.";

   // Check that the serialized file was created
   ASSERT_TRUE(fileExists(serialFull)) << "Serialized full simulation file does not exist.";
}
