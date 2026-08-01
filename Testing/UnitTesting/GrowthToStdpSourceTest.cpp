/**
 * @file GrowthToStdpSourceTest.cpp
 *
 * @brief First stage of the growth-to-STDP integration test: grow a network and serialize it.
 *
 *    STAGE 1: Run a short growth simulation (ConnGrowth with AllDSSynapses) and save its
 *             serialized output. Reference file: /Testing/RegressionTesting/configfiles/
 *             test-growth-stdp-source.xml
 *    STAGE 2: Run an STDP simulation (ConnStatic with AllSTDPSynapses) that deserializes the
 *             checkpoint from stage 1, and verify that the grown topology was imported into the
 *             STDP network. See GrowthToStdpImportTest.cpp.
 *
 * @note This test covers stage 1. The checkpoint it writes is the input to stage 2, so the two
 *       executables must run in order; `run_growth_stdp_test.sh` in the `build` directory does
 *       that.
 *
 * @ingroup Testing/UnitTesting
 */

#include "GrowthToStdpHelper.cpp"
#include "gtest/gtest.h"

using namespace std;

// Run the growth simulation that produces the input network for the STDP simulation
TEST(GrowthToStdpTest, SerializeGrownNetwork)
{
   string executable = "./cgraphitti";

   // Configuration file for the growth simulation
   string configFile = "../Testing/RegressionTesting/configfiles/test-growth-stdp-source.xml";

   // Path to save the serialized grown network
   string growthCheckpoint = "../Testing/UnitTesting/TestOutput/growth-network-checkpoint.xml";

   // Command-line arguments for the simulation
   string arguments = "-c " + configFile + " -s " + growthCheckpoint;

   // Run simulation
   ASSERT_TRUE(runSimulation(executable, arguments)) << "Growth simulation failed.";

   // Check that the serialized file was created
   ASSERT_TRUE(fileExists(growthCheckpoint)) << "Growth checkpoint file does not exist.";

   CheckpointInfo grown;
   ASSERT_TRUE(readCheckpointInfo(growthCheckpoint, grown)) << "Could not read growth checkpoint.";

   EXPECT_EQ("ConnGrowth", grown.connectionsClass);

   // Stage 2 has nothing to import unless the growth simulation actually grew edges, so guard
   // against a configuration change that would quietly make this test vacuous.
   EXPECT_GT(grown.totalEdgeCount, 0) << "Growth simulation produced no edges to import.";
}
