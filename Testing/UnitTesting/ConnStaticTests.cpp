/**
 * @file ConnStaticTests.cpp
 *
 * @brief Unit tests for ConnStatic.
 *
 * @ingroup Testing/UnitTesting
 */

#include "AllSTDPSynapses.h"
#include "ConnStatic.h"
#include "Model.h"
#include "ParameterManager.h"
#include "XmlRecorder.h"
#include "gtest/gtest.h"
#include <memory>
#include <vector>

namespace {

   class TestSTDPSynapses : public AllSTDPSynapses {
   public:
      using AllSTDPSynapses::AllSTDPSynapses;

      void setWeight(BGSIZE index, BGFLOAT weight)
      {
         W_[index] = weight;
      }
   };

   class TestConnStatic : public ConnStatic {
   public:
      void setEdges(std::unique_ptr<AllEdges> edges)
      {
         edges_ = std::move(edges);
      }
   };

   TEST(ConnStatic, UpdateConnectionsCopiesActiveEdges)
   {
      auto edges = std::make_unique<AllSTDPSynapses>(3, 2);
      edges->addEdge(edgeType::EE, 0, 2, 0.001);
      edges->addEdge(edgeType::II, 2, 0, 0.001);

      const std::vector<int> expectedSources {2, 0};
      const std::vector<int> expectedDestinations {0, 2};
      const std::vector<BGFLOAT> expectedWeights {-10.0e-9, 10.0e-9};

      TestConnStatic connections;
      connections.setEdges(std::move(edges));

      connections.updateConnections();

      EXPECT_EQ(expectedSources, connections.getSourceVertexIndexCurrentEpoch());
      EXPECT_EQ(expectedDestinations, connections.getDestVertexIndexCurrentEpoch());
      EXPECT_EQ(expectedWeights, connections.getWCurrentEpoch());
   }

   TEST(ConnStatic, RecorderCapturesThreeEpochsWithoutDuplicateEdges)
   {
      ASSERT_TRUE(ParameterManager::getInstance().loadParameterFile(
         "../Testing/RegressionTesting/configfiles/test-tiny.xml"));
      ASSERT_TRUE(Simulator::getInstance().instantiateSimulatorObjects());
      auto *recorder
         = dynamic_cast<XmlRecorder *>(&Simulator::getInstance().getModel().getRecorder());
      ASSERT_NE(nullptr, recorder);

      auto edges = std::make_unique<TestSTDPSynapses>(3, 2);
      const BGSIZE changedEdge = edges->addEdge(edgeType::EE, 0, 2, 0.001);
      edges->addEdge(edgeType::II, 2, 0, 0.001);
      TestSTDPSynapses &edgeData = *edges;

      TestConnStatic connections;
      connections.setEdges(std::move(edges));
      connections.registerHistoryVariables();

      std::vector<multipleTypes> recordedWeights;
      std::vector<multipleTypes> recordedSources;
      std::vector<multipleTypes> recordedDestinations;
      for (int epoch = 0; epoch < 3; ++epoch) {
         SCOPED_TRACE(epoch);
         edgeData.setWeight(changedEdge, BGFLOAT(epoch + 1) * 10.0e-9);

         const AllEdges &allEdges = connections.getEdges();
         std::vector<BGFLOAT> expectedWeights;
         std::vector<int> expectedSources;
         std::vector<int> expectedDestinations;
         for (size_t slot = 0; slot < allEdges.getInUse().size(); ++slot) {
            if (allEdges.getInUse()[slot]) {
               expectedWeights.push_back(allEdges.getWeights()[slot]);
               expectedSources.push_back(allEdges.getSourceVertexIndices()[slot]);
               expectedDestinations.push_back(allEdges.getDestVertexIndices()[slot]);
            }
         }
         ASSERT_FALSE(expectedWeights.empty());
         ASSERT_LT(expectedWeights.size(), allEdges.getInUse().size());

         EXPECT_FALSE(connections.updateConnections());
         ASSERT_EQ(expectedWeights.size(), connections.getWCurrentEpoch().size());
         ASSERT_EQ(connections.getWCurrentEpoch().size(),
                   connections.getSourceVertexIndexCurrentEpoch().size());
         ASSERT_EQ(connections.getWCurrentEpoch().size(),
                   connections.getDestVertexIndexCurrentEpoch().size());
         EXPECT_EQ(expectedWeights, connections.getWCurrentEpoch());
         EXPECT_EQ(expectedSources, connections.getSourceVertexIndexCurrentEpoch());
         EXPECT_EQ(expectedDestinations, connections.getDestVertexIndexCurrentEpoch());

         for (size_t i = 0; i < expectedWeights.size(); ++i) {
            recordedWeights.push_back(expectedWeights[i]);
            recordedSources.push_back(expectedSources[i]);
            recordedDestinations.push_back(expectedDestinations[i]);
         }
         recorder->compileHistories();
         EXPECT_TRUE(connections.getWCurrentEpoch().empty());
         EXPECT_TRUE(connections.getSourceVertexIndexCurrentEpoch().empty());
         EXPECT_TRUE(connections.getDestVertexIndexCurrentEpoch().empty());
         EXPECT_EQ(recordedWeights, recorder->getHistory(0));
         EXPECT_EQ(recordedSources, recorder->getHistory(1));
         EXPECT_EQ(recordedDestinations, recorder->getHistory(2));
      }
   }

}   // namespace
