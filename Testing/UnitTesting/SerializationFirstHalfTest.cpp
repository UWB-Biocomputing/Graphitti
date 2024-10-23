/**
 * @file SerializationFirstHalfTest.cpp
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
 * @note This specific test, SerializationFirstHalfTest, covers the third step: running the first half of the simulation and saving its serialized output
 * 
 * To run the serialization tests, execute the script `run_serial_test.sh` located in the `build` directory.
 * 
 * @ingroup Testing/UnitTesting
 */


#include "SerializationHelper.cpp"
#include "gtest/gtest.h"

using namespace std;

// Test serialization by running full and half simulations, and comparing serialized files
TEST(SerializationTest, SerializeFirstHalfTest)
{
   string executable = "./cgraphitti";

   // Configuration file for the half simulation
   string configFileHalf = "../configfiles/test-small-long-half.xml";

   // Path to save the serialized output file
   string serialFirstHalf = "../Testing/UnitTesting/TestOutput/First-half-serialized-file.xml";

   // Command-line arguments for the simulation
   string argumentFirstHalf = "-c " + configFileHalf + " -s " + serialFirstHalf;

   // Run simulations
   ASSERT_TRUE(runSimulation(executable, argumentFirstHalf)) << "First half simulation failed.";

   // Check that the serialized file was created
   ASSERT_TRUE(fileExists(serialFirstHalf)) << "Serialized first half file does not exist.";
}