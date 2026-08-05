/**
 * @file ConnStaticTests.cpp
 *
 * @brief Unit tests for ConnStatic.
 *
 * @ingroup Testing/UnitTesting
 */

#include "AllSTDPSynapses.h"
#include "ConnStatic.h"
#include "gtest/gtest.h"
#include <memory>
#include <vector>

namespace {

   class TestConnStatic : public ConnStatic {
   public:
      void setEdges(std::unique_ptr<AllEdges> edges)
      {
         edges_ = std::move(edges);
      }
   };

   TEST(ConnStatic, UpdateConnectionsCopiesActiveEdgesWithoutAccumulating)
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

      connections.updateConnections();

      EXPECT_EQ(expectedSources, connections.getSourceVertexIndexCurrentEpoch());
      EXPECT_EQ(expectedDestinations, connections.getDestVertexIndexCurrentEpoch());
      EXPECT_EQ(expectedWeights, connections.getWCurrentEpoch());
   }

}   // namespace
