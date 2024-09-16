/**
 * @file SerializationSecondHalfTest.cpp
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
 * @note This specific test, SerializationSecondHalfTest, covers the fourth and fifth step above.
 * 
 * To run the serialization tests, execute the script `run_serial_test.sh` located in the `build` directory.
 *
 * @ingroup Testing/UnitTesting
 */

#include "SerializationHelper.cpp"
#include "gtest/gtest.h"

using namespace std;

// Test serialization by running full and half simulations, and comparing serialized files
TEST(SerializationSecondHalf, SerializeSecondHalfFileTest)
{
   string executable = "./cgraphitti";

   // Configuration file for the half simulation
   string configFileHalf = "../configfiles/test-small-long-half.xml";

   // Path to saved serialized output file
   string serialFull = "../Testing/UnitTesting/TestOutput/Full-serialized-file.xml";
   string serialFirstHalf = "../Testing/UnitTesting/TestOutput/First-half-serialized-file.xml";
   string serialSecondHalf = "../Testing/UnitTesting/TestOutput/Second-half-serialized-file.xml";

   // Command-line arguments for the simulation
   string argumentSecondHalf
      = "-c " + configFileHalf + " -d " + serialFirstHalf + " -s " + serialSecondHalf;

   // Run simulations
   ASSERT_TRUE(runSimulation(executable, argumentSecondHalf)) << "Second half simulation failed.";

   // Check if serialized files exist
   ASSERT_TRUE(fileExists(serialSecondHalf)) << "Serialized second half file does not exist.";

   // Compare the full serialized file with the second-half serialized file
   ASSERT_TRUE(compareXmlFiles(serialFull, serialSecondHalf)) << "Serialized files do not match.";
}
