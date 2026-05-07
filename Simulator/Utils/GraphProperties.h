#pragma once

#include <string>

/// @brief Parent structure to store common properties for all graph vertices
struct VertexProperties {
   std::string type;
   double x = 0.0;
   double y = 0.0;
};

/// @brief Derived structure for NG911-specific properties
/// Inherits from VertexProperty and includes attributes specific to 911 networks
struct NG911VertexProperties : public VertexProperties {
   std::string objectID;
   std::string name;
   int servers = 0;
   int trunks = 0;
   std::string segments;
};

/// @brief Derived structure for Neural Network-specific properties
struct NeuralVertexProperties : public VertexProperties {
   bool active = false;
};

/// @brief The structure to hold the edge properties
struct NeuralEdgeProperties {
   int source;
   int target;
   double weight;
};

/// @brief The structure to hold the Graph properties
struct GraphProperties {
   // TODO: Graph Properties
};
