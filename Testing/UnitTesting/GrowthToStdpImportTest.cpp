/**
 * @file GrowthToStdpImportTest.cpp
 *
 * @brief Second stage of the growth-to-STDP integration test: start an STDP simulation from a
 *        serialized growth network.
 *
 *    STAGE 1: Run a short growth simulation (ConnGrowth with AllDSSynapses) and save its
 *             serialized output. See GrowthToStdpSourceTest.cpp.
 *    STAGE 2: Run an STDP simulation (ConnStatic with AllSTDPSynapses) that deserializes the
 *             checkpoint from stage 1, and verify that the grown topology was imported into the
 *             STDP network. Reference file: /Testing/RegressionTesting/configfiles/
 *             test-growth-stdp.xml
 *
 * @note This test covers stage 2 and consumes the checkpoint written by stage 1, so the two
 *       executables must run in order; `run_growth_stdp_test.sh` in the `build` directory does
 *       that.
 *
 * The STDP simulation is asked to serialize its own final state as well. Comparing the two
 * checkpoints is what makes the import observable: the classes must be the ones named in the STDP
 * configuration file, while the edge count must be the one carried over from the growth run.
 *
 * @ingroup Testing/UnitTesting
 */

#include "GrowthToStdpHelper.cpp"
#include "gtest/gtest.h"

using namespace std;

// Deserialize a grown network into an STDP simulation and verify the topology was imported
TEST(GrowthToStdpTest, ImportGrownNetworkIntoStdpSimulation)
{
   string executable = "./cgraphitti";

   // Configuration file for the STDP simulation
   string configFile = "../Testing/RegressionTesting/configfiles/test-growth-stdp.xml";

   // Serialized grown network written by GrowthToStdpSourceTest
   string growthCheckpoint = "../Testing/UnitTesting/TestOutput/growth-network-checkpoint.xml";

   // Path to save the serialized output of the STDP simulation
   string stdpCheckpoint = "../Testing/UnitTesting/TestOutput/stdp-from-growth-checkpoint.xml";

   ASSERT_TRUE(fileExists(growthCheckpoint))
      << "Growth checkpoint does not exist. Run growthStdpSourceTest first.";

   CheckpointInfo grown;
   ASSERT_TRUE(readCheckpointInfo(growthCheckpoint, grown)) << "Could not read growth checkpoint.";
   ASSERT_GT(grown.totalEdgeCount, 0) << "Growth checkpoint contains no edges to import.";

   // Command-line arguments for the simulation
   string arguments = "-c " + configFile + " -d " + growthCheckpoint + " -s " + stdpCheckpoint;

   // Run simulation
   ASSERT_TRUE(runSimulation(executable, arguments))
      << "STDP simulation from the grown network failed.";

   // Check that the serialized file was created
   ASSERT_TRUE(fileExists(stdpCheckpoint)) << "STDP checkpoint file does not exist.";

   CheckpointInfo imported;
   ASSERT_TRUE(readCheckpointInfo(stdpCheckpoint, imported)) << "Could not read STDP checkpoint.";

   // Deserialization restores whatever classes the checkpoint holds, which used to overwrite the
   // classes named in the configuration file and silently continue the growth simulation.
   EXPECT_EQ("ConnStatic", imported.connectionsClass);
   EXPECT_EQ("AllSTDPSynapses", imported.edgesClass);

   // The STDP network's edges come from the grown topology rather than from its own (edge-free)
   // graph file, so every grown edge must survive the import.
   EXPECT_EQ(grown.totalEdgeCount, imported.totalEdgeCount);
}
