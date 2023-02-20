/**
* @file Connections911.cpp
*
* @ingroup Simulator/Connections/NG911
* 
* @brief The model of the static network
*
*/

#include "Connections911.h"
#include "All911Vertices.h"
#include "GraphManager.h"
#include "Layout911.h"
#include "ParameterManager.h"

Connections911::Connections911()
{
}

Connections911::~Connections911()
{
   if (oldTypeMap_ != nullptr)
      delete[] oldTypeMap_;
}

void Connections911::setup()
{
   int added = 0;
   LOG4CPLUS_INFO(fileLogger_, "Initializing connections");

   // we can obtain the Layout, which holds the vertices, from the Model
   shared_ptr<Layout> layout = Simulator::getInstance().getModel()->getLayout();
   shared_ptr<AllVertices> vertices = layout->getVertices();

   // Get list of edges sorted by target in ascending order from GraphManager
   GraphManager &gm = GraphManager::getInstance();
   auto sorted_edge_list = gm.edgesSortByTarget();

   // add sorted edges
   for (auto it = sorted_edge_list.begin(); it != sorted_edge_list.end(); ++it) {
      size_t srcV = gm.source(*it);
      size_t destV = gm.target(*it);
      edgeType type = layout->edgType(srcV, destV);
      BGFLOAT *sumPoint = &vertices->summationMap_[destV];

      BGFLOAT dist = (*layout->dist_)(srcV, destV);
      LOG4CPLUS_DEBUG(edgeLogger_, "Source: " << srcV << " Dest: " << destV << " Dist: " << dist);

      BGSIZE iEdg;
      edges_->addEdge(iEdg, type, srcV, destV, sumPoint, Simulator::getInstance().getDeltaT());
      added++;
   }

   LOG4CPLUS_DEBUG(fileLogger_, "Added connections: " << added);
}

void Connections911::loadParameters()
{
   ParameterManager::getInstance().getIntByXpath("//connsPerVertex/text()", connsPerVertex_);
   ParameterManager::getInstance().getIntByXpath("//psapsToErase/text()", psapsToErase_);
   ParameterManager::getInstance().getIntByXpath("//respsToErase/text()", respsToErase_);
}

///  Prints out all parameters to logging file.
///  Registered to OperationManager as Operation::printParameters
void Connections911::printParameters() const
{
   LOG4CPLUS_DEBUG(fileLogger_, "CONNECTIONS PARAMETERS"
                                   << endl
                                   << "\tConnections Type: Connections911" << endl
                                   << "\tConnections per vertex: " << connsPerVertex_ << endl
                                   << "\tPSAPs to erase: " << psapsToErase_ << endl
                                   << "\tRESPs to erase: " << respsToErase_ << endl
                                   << endl);
}

#if !defined(USE_GPU)
///  Update the connections status in every epoch.
///
///  @param  vertices  The Vertex list to search from.
///  @return true if successful, false otherwise.
bool Connections911::updateConnections(AllVertices &vertices)
{
   // Only run on the first epoch
   if (Simulator::getInstance().getCurrentStep() != 1) {
      return false;
   }

   // Record old type map
   int numVertices = Simulator::getInstance().getTotalVertices();
   Layout &layout = *Simulator::getInstance().getModel()->getLayout();
   oldTypeMap_ = new vertexType[numVertices];
   memcpy(oldTypeMap_, layout.vertexTypeMap_, numVertices * sizeof(vertexType));

   // Erase PSAPs
   for (int i = 0; i < psapsToErase_; i++) {
      erasePSAP(vertices, layout);
   }

   // Erase RESPs
   for (int i = 0; i < respsToErase_; i++) {
      eraseRESP(vertices, layout);
   }

   return true;
}

///  Randomly delete 1 PSAP and rewire all the edges around it.
///
///  @param  vertices  The Vertex list to search from.
///  @param  layout   Layout information of the vertex network.
///  @return true if successful, false otherwise.
bool Connections911::erasePSAP(AllVertices &vertices, Layout &layout)
{
   int numVertices = Simulator::getInstance().getTotalVertices();

   vector<int> psaps;
   psaps.clear();

   // Find all psaps
   for (int i = 0; i < numVertices; i++) {
      if (layout.vertexTypeMap_[i] == PSAP) {
         psaps.push_back(i);
      }
   }

   // Only 1 psap, do not delete me :(
   if (psaps.size() < 2) {
      return false;
   }

   // Pick random PSAP
   int randVal = initRNG.inRange(0, psaps.size());
   int randPSAP = psaps[randVal];
   psaps.erase(psaps.begin() + randVal);

   BGSIZE maxTotalEdges = edges_->maxEdgesPerVertex_ * numVertices;
   bool changesMade = false;
   vector<int> callersToReroute;
   vector<int> respsToReroute;

   callersToReroute.clear();
   respsToReroute.clear();

   // Iterate through all edges
   for (int iEdg = 0; iEdg < maxTotalEdges; iEdg++) {
      if (!edges_->inUse_[iEdg]) {
         continue;
      }
      int srcVertex = edges_->sourceVertexIndex_[iEdg];
      int destVertex = edges_->destVertexIndex_[iEdg];

      // Find PSAP edge
      if (srcVertex == randPSAP || destVertex == randPSAP) {
         // Record erased edge
         ChangedEdge erasedEdge;
         erasedEdge.srcV = srcVertex;
         erasedEdge.destV = destVertex;
         erasedEdge.eType = layout.edgType(srcVertex, destVertex);
         edgesErased.push_back(erasedEdge);

         changesMade = true;
         edges_->eraseEdge(destVertex, iEdg);

         // Identify all psap-less callers
         if (layout.vertexTypeMap_[srcVertex] == CALR) {
            callersToReroute.push_back(srcVertex);
         }

         // Identify all psap-less responders
         if (layout.vertexTypeMap_[destVertex] == RESP) {
            respsToReroute.push_back(destVertex);
         }
      }
   }

   if (changesMade) {
      // This is here so that we don't delete the vertex if we can't find any edges
      verticesErased.push_back(randPSAP);
      layout.vertexTypeMap_[randPSAP] = VTYPE_UNDEF;
   }

   // Failsafe
   if (psaps.size() < 1) {
      return false;
   }

   // For each psap-less caller, find closest match
   for (int i = 0; i < callersToReroute.size(); i++) {
      int srcVertex = callersToReroute[i];

      int closestPSAP = psaps[0];
      BGFLOAT smallestDist = (*layout.dist_)(srcVertex, closestPSAP);

      // Find closest PSAP
      for (int i = 0; i < psaps.size(); i++) {
         BGFLOAT dist = (*layout.dist_)(srcVertex, psaps[i]);
         if (dist < smallestDist) {
            smallestDist = dist;
            closestPSAP = psaps[i];
         }
      }

      // Insert Caller to PSAP edge
      BGFLOAT *sumPoint = &vertices.summationMap_[closestPSAP];
      BGSIZE iEdg;
      edges_->addEdge(iEdg, CP, srcVertex, closestPSAP, sumPoint,
                      Simulator::getInstance().getDeltaT());

      // Record added edge
      ChangedEdge addedEdge;
      addedEdge.srcV = srcVertex;
      addedEdge.destV = closestPSAP;
      addedEdge.eType = CP;
      edgesAdded.push_back(addedEdge);
   }

   // For each psap-less responder, find closest match
   for (int i = 0; i < respsToReroute.size(); i++) {
      int destVertex = respsToReroute[i];

      int closestPSAP = psaps[0];
      BGFLOAT smallestDist = (*layout.dist_)(closestPSAP, destVertex);

      // Find closest PSAP
      for (int i = 0; i < psaps.size(); i++) {
         BGFLOAT dist = (*layout.dist_)(psaps[i], destVertex);
         if (dist < smallestDist) {
            smallestDist = dist;
            closestPSAP = psaps[i];
         }
      }

      // Insert PSAP to Responder edge
      BGFLOAT *sumPoint = &vertices.summationMap_[destVertex];
      BGSIZE iEdg;
      edges_->addEdge(iEdg, PR, closestPSAP, destVertex, sumPoint,
                      Simulator::getInstance().getDeltaT());

      // Record added edge
      ChangedEdge addedEdge;
      addedEdge.srcV = closestPSAP;
      addedEdge.destV = destVertex;
      addedEdge.eType = PR;
      edgesAdded.push_back(addedEdge);
   }

   return changesMade;
}

///  Randomly delete 1 RESP.
///
///  @param  vertices  The Vertex list to search from.
///  @param  layout   Layout information of the vertex network.
///  @return true if successful, false otherwise.
bool Connections911::eraseRESP(AllVertices &vertices, Layout &layout)
{
   int numVertices = Simulator::getInstance().getTotalVertices();

   vector<int> resps;
   resps.clear();

   // Find all resps
   for (int i = 0; i < numVertices; i++) {
      if (layout.vertexTypeMap_[i] == RESP) {
         resps.push_back(i);
      }
   }

   // Only 1 resp, do not delete me :(
   if (resps.size() < 2) {
      return false;
   }

   // Pick random RESP
   int randVal = initRNG.inRange(0, resps.size());
   int randRESP = resps[randVal];
   resps.erase(resps.begin() + randVal);

   BGSIZE maxTotalEdges = edges_->maxEdgesPerVertex_ * numVertices;
   bool changesMade = false;

   // Iterate through all edges
   for (int iEdg = 0; iEdg < maxTotalEdges; iEdg++) {
      if (!edges_->inUse_[iEdg]) {
         continue;
      }
      int srcVertex = edges_->sourceVertexIndex_[iEdg];
      int destVertex = edges_->destVertexIndex_[iEdg];

      // Find RESP edge
      if (srcVertex == randRESP || destVertex == randRESP) {
         // Record erased edge
         ChangedEdge erasedEdge;
         erasedEdge.srcV = srcVertex;
         erasedEdge.destV = destVertex;
         erasedEdge.eType = layout.edgType(srcVertex, destVertex);
         edgesErased.push_back(erasedEdge);

         changesMade = true;
         edges_->eraseEdge(destVertex, iEdg);
      }
   }

   if (changesMade) {
      // This is here so that we don't delete the vertex if we can't find any edges
      verticesErased.push_back(randRESP);
      layout.vertexTypeMap_[randRESP] = VTYPE_UNDEF;
   }

   return changesMade;
}

///  @return xml representation of a single edge
string Connections911::ChangedEdge::toString()
{
   stringstream os;
   string type_s;

   switch (eType) {
      case CP:
         type_s = "CP";
         break;
      case PR:
         type_s = "PR";
         break;
      case PP:
         type_s = "PP";
         break;
      case PC:
         type_s = "PC";
         break;
      case RP:
         type_s = "RP";
         break;
      case RC:
         type_s = "RC";
         break;
      default:
         type_s = "ETYPE_UNDEF";
   }

   os << "<item>";
   os << srcV << " " << destV << " " << type_s;
   os << "</item>" << endl;

   return os.str();
}

///  Returns the complete list of all deleted or added edges as a string.
///  @param added    true returns the list of added edges, false = erased
///  @return xml representation of all deleted or added edges
string Connections911::changedEdgesToXML(bool added)
{
   stringstream os;

   vector<ChangedEdge> changed = edgesErased;
   string name = "edgesDeleted";

   if (added) {
      changed = edgesAdded;
      name = "edgesAdded";
   }

   os << "<Matrix name=\"" << name << "\" type=\"complete\" rows=\"" << changed.size()
      << "\" columns=\"3\" multiplier=\"1.0\">" << endl;

   for (int i = 0; i < changed.size(); i++) {
      os << "   " << changed[i].toString();
   }

   os << "</Matrix>";
   return os.str();
}

///  Returns the complete list of deleted vertices as a string.
///  @return xml representation of all deleted vertices
string Connections911::erasedVerticesToXML()
{
   stringstream os;

   os << "<Matrix name=\"verticesDeleted\" type=\"complete\" rows=\"1\" columns=\""
      << verticesErased.size() << "\" multiplier=\"1.0\">" << endl;
   os << "   ";

   sort(verticesErased.begin(), verticesErased.end());
   for (int i = 0; i < verticesErased.size(); i++) {
      os << verticesErased[i] << " ";
   }

   os << endl << "</Matrix>";
   return os.str();
}

#endif