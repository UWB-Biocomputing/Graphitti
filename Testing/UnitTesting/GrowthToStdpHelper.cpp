/**
 * @file GrowthToStdpHelper.cpp
 *
 * @brief Helper functions shared by the growth-to-STDP integration tests.
 *
 * These tests verify that the network produced by a growth simulation can be used as the input
 * network for a subsequent STDP simulation (see Serializer::deserialize()). Both stages assert on
 * the contents of Cereal checkpoint files, so the helper here extracts the connection class, the
 * edge class, and the active edge count from a checkpoint.
 *
 * @ingroup Testing/UnitTesting
 */

#include "SerializationHelper.cpp"

using namespace std;

/// The parts of a Cereal checkpoint that the growth-to-STDP tests assert on.
struct CheckpointInfo {
   string connectionsClass;
   string edgesClass;
   int totalEdgeCount = -1;
};

namespace {

   /// Recursively finds the first descendant element with the given name.
   TiXmlElement *findFirstElement(TiXmlElement *parent, const string &name)
   {
      if (!parent) {
         return nullptr;
      }
      for (TiXmlElement *child = parent->FirstChildElement(); child;
           child = child->NextSiblingElement()) {
         if (name == child->Value()) {
            return child;
         }
         if (TiXmlElement *found = findFirstElement(child, name)) {
            return found;
         }
      }
      return nullptr;
   }

   /// Returns the text of a direct child element, or an empty string when it is absent.
   string childText(TiXmlElement *parent, const string &name)
   {
      if (!parent) {
         return "";
      }
      TiXmlElement *child = parent->FirstChildElement(name.c_str());
      const char *text = child ? child->GetText() : nullptr;
      return text ? text : "";
   }

}   // namespace

/// Reads the connection class, edge class, and active edge count from a Cereal checkpoint.
///
/// Checkpoints nest the edges inside the connections, so the edge lookups are scoped to the
/// connections subtree instead of searching the whole document.
///
/// @param path  Path to a checkpoint written by Graphitti's `-s` option.
/// @param info  Populated when every field is found.
/// @return true on success, false if the file cannot be read or a field is missing.
bool readCheckpointInfo(const string &path, CheckpointInfo &info)
{
   TiXmlDocument document;
   if (!document.LoadFile(path.c_str())) {
      cerr << "Failed to load checkpoint file: " << path << endl;
      return false;
   }

   TiXmlElement *connections = findFirstElement(document.RootElement(), "connections");
   if (!connections) {
      cerr << "Checkpoint has no connections element: " << path << endl;
      return false;
   }
   info.connectionsClass = childText(connections, "polymorphic_name");

   TiXmlElement *edges = findFirstElement(connections, "edges");
   if (!edges) {
      cerr << "Checkpoint has no edges element: " << path << endl;
      return false;
   }
   info.edgesClass = childText(edges, "polymorphic_name");

   TiXmlElement *edgeCount = findFirstElement(edges, "totalEdgeCount");
   const char *countText = edgeCount ? edgeCount->GetText() : nullptr;
   if (!countText) {
      cerr << "Checkpoint has no totalEdgeCount element: " << path << endl;
      return false;
   }
   try {
      info.totalEdgeCount = stoi(countText);
   } catch (const exception &e) {
      cerr << "Could not parse totalEdgeCount in " << path << ": " << e.what() << endl;
      return false;
   }

   return !info.connectionsClass.empty() && !info.edgesClass.empty();
}
