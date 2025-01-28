/**
 * @file AllNeuroEdges.cpp
 * 
 * @ingroup Simulator/Edges
 *
 * @brief
 */

#include "AllNeuroEdges.h"

///  Setup the internal structure of the class (allocate memories and initialize them).
void AllNeuroEdges::setupEdges()
{
   int maxEdges = Simulator::getInstance().getMaxEdgesPerVertex();
   int numVertices = Simulator::getInstance().getTotalVertices();
   setupEdges(numVertices, maxEdges);
}

///  Setup the internal structure of the class (allocate memories and initialize them).
void AllNeuroEdges::setupEdges(int numVertices, int maxEdges)
{
   AllEdges::setupEdges(numVertices, maxEdges);

   BGSIZE maxTotalEdges = maxEdges * numVertices;

   if (maxTotalEdges != 0) {
      psr_.assign(maxTotalEdges, 0.0);
   }
}

///  Reset time varying state vars and recompute decay.
///
///  @param  iEdg     Index of the edge to set.
///  @param  deltaT   Inner simulation step duration
void AllNeuroEdges::resetEdge(BGSIZE iEdg, BGFLOAT deltaT)
{
   psr_[iEdg] = 0.0;
}

///  Sets the data for Synapse to input's data.
///
///  @param  input  istream to read from.
///  @param  iEdg   Index of the edge to set.
void AllNeuroEdges::readEdge(istream &input, BGSIZE iEdg)
{
   int synapse_type(0);

   // input.ignore() so input skips over end-of-line characters.
   input >> sourceVertexIndex_[iEdg];
   input.ignore();
   input >> destVertexIndex_[iEdg];
   input.ignore();
   input >> W_[iEdg];
   input.ignore();
   input >> psr_[iEdg];
   input.ignore();
   input >> synapse_type;
   input.ignore();
   input >> inUse_[iEdg];
   input.ignore();

   type_[iEdg] = edgeOrdinalToType(synapse_type);
}

///  Write the edge data to the stream.
///
///  @param  output  stream to print out to.
///  @param  iEdg    Index of the edge to print out.
void AllNeuroEdges::writeEdge(ostream &output, BGSIZE iEdg) const
{
   output << sourceVertexIndex_[iEdg] << ends;
   output << destVertexIndex_[iEdg] << ends;
   output << W_[iEdg] << ends;
   output << psr_[iEdg] << ends;
   output << static_cast<int>(type_[iEdg]) << ends;
   output << (inUse_[iEdg] == 1 ? "true" : "false") << ends;
}

///  Get the sign of the edgeType.
///
///  @param    type    edgeType I to I, I to E, E to I, or E to E
///  @return   1 or -1, or 0 if error
int AllNeuroEdges::edgSign(const edgeType type)
{
   switch (type) {
      case edgeType::II:
      case edgeType::IE:
         return -1;
      case edgeType::EI:
      case edgeType::EE:
         return 1;
      case edgeType::ETYPE_UNDEF:
         return 0;
      default:
         return 0;
   }
   // TODO: exception throw.
   return 0;   // default.
}

///  Prints SynapsesProps data to console.
void AllNeuroEdges::printSynapsesProps() const
{
   cout << "This is SynapsesProps data:" << endl;
   for (int i = 0; i < maxEdgesPerVertex_ * countVertices_; i++) {
      if (W_[i] != 0.0) {
         cout << "W[" << i << "] = " << W_[i];
         cout << " sourNeuron: " << sourceVertexIndex_[i];
         cout << " desNeuron: " << destVertexIndex_[i];
         cout << " type: " << static_cast<int>(type_[i]);
         cout << " psr: " << psr_[i];
         cout << " in_use:" << (inUse_[i] == 1 ? "true" : "false");
      }
   }

   for (int i = 0; i < countVertices_; i++) {
      cout << "edge_counts:" << "vertex[" << i << "]" << edgeCounts_[i] << endl;
   }

   cout << "totalEdgeCount:" << totalEdgeCount_ << endl;
   cout << "maxEdgesPerVertex:" << maxEdgesPerVertex_ << endl;
   cout << "count_neurons:" << countVertices_ << endl;
}
