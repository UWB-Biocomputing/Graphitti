/**
 * @file AllEdges.cpp
 * 
 * @ingroup Simulator/Edges
 *
 * @brief
 */

#include "AllEdges.h"
#include "AllVertices.h"
#include "OperationManager.h"

AllEdges::AllEdges() :
      totalEdgeCount_(0),
      maxEdgesPerVertex_(0),
      countVertices_(0) {
   destNeuronIndex_ = nullptr;
   W_ = nullptr;
   summationPoint_ = nullptr;
   sourceNeuronIndex_ = nullptr;
   psr_ = nullptr;
   type_ = nullptr;
   inUse_ = nullptr;
   synapseCounts_ = nullptr;

   // Register loadParameters function as a loadParameters operation in the OperationManager
   function<void()> loadParametersFunc = std::bind(&IAllEdges::loadParameters, this);
   OperationManager::getInstance().registerOperation(Operations::op::loadParameters, loadParametersFunc);

   // Register printParameters function as a printParameters operation in the OperationManager
   function<void()> printParametersFunc = bind(&IAllEdges::printParameters, this);
   OperationManager::getInstance().registerOperation(Operations::printParameters, printParametersFunc);

   fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));
}

AllEdges::AllEdges(const int numVertices, const int maxEdges) {
   setupEdges(numVertices, maxEdges);
}

AllEdges::~AllEdges() {
   BGSIZE maxTotalSynapses = maxEdgesPerVertex_ * countVertices_;

  if (maxTotalSynapses != 0) {
     delete[] destNeuronIndex_;
     delete[] W_;
     delete[] summationPoint_;
     delete[] sourceNeuronIndex_;
     delete[] psr_;
     delete[] type_;
     delete[] inUse_;
     delete[] synapseCounts_;
  }

   destNeuronIndex_ = nullptr;
   W_ = nullptr;
   summationPoint_ = nullptr;
   sourceNeuronIndex_ = nullptr;
   psr_ = nullptr;
   type_ = nullptr;
   inUse_ = nullptr;
   synapseCounts_ = nullptr;

   countVertices_ = 0;
   maxEdgesPerVertex_ = 0;
}

///  Setup the internal structure of the class (allocate memories and initialize them).
void AllEdges::setupEdges() {
   setupEdges(Simulator::getInstance().getTotalVertices(), Simulator::getInstance().getMaxEdgesPerVertex());
}

///  Setup the internal structure of the class (allocate memories and initialize them).
///
///  @param  numVertices   Total number of vertices in the network.
///  @param  max_synapses  Maximum number of edges per vertex.
void AllEdges::setupEdges(const int numVertices, const int maxEdges) {
   BGSIZE maxTotalSynapses = maxEdges * numVertices;

   maxEdgesPerVertex_ = maxEdges;
   totalEdgeCount_ = 0;
   countVertices_ = numVertices;

   if (maxTotalSynapses != 0) {
      destNeuronIndex_ = new int[maxTotalSynapses];
      W_ = new BGFLOAT[maxTotalSynapses];
      summationPoint_ = new BGFLOAT *[maxTotalSynapses];
      sourceNeuronIndex_ = new int[maxTotalSynapses];
      psr_ = new BGFLOAT[maxTotalSynapses];
      type_ = new synapseType[maxTotalSynapses];
      inUse_ = new bool[maxTotalSynapses];
      synapseCounts_ = new BGSIZE[numVertices];

      for (BGSIZE i = 0; i < maxTotalSynapses; i++) {
         summationPoint_[i] = nullptr;
         inUse_[i] = false;
         W_[i] = 0;
      }

      for (int i = 0; i < numVertices; i++) {
         synapseCounts_[i] = 0;
      }
   }
}

/// Load member variables from configuration file.
/// Registered to OperationManager as Operation::op::loadParameters
void AllEdges::loadParameters() {
   // Nothing to load from configuration file besides SynapseType in the current implementation.
}

///  Prints out all parameters to logging file.
///  Registered to OperationManager as Operation::printParameters
void AllEdges::printParameters() const {
   LOG4CPLUS_DEBUG(fileLogger_, "\nEDGES PARAMETERS" << endl
    << "\t---AllEdges Parameters---" << endl
    << "\tTotal synapse counts: " << totalEdgeCount_ << endl
    << "\tMax edges per vertex: " << maxEdgesPerVertex_ << endl
    << "\tNeuron count: " << countVertices_ << endl << endl);
}

///  Reset time varying state vars and recompute decay.
///
///  @param  iEdg     Index of the synapse to set.
///  @param  deltaT   Inner simulation step duration
void AllEdges::resetEdge(const BGSIZE iEdg, const BGFLOAT deltaT) {
   psr_[iEdg] = 0.0;
}

///  Sets the data for Synapse to input's data.
///
///  @param  input  istream to read from.
///  @param  iEdg   Index of the synapse to set.
void AllEdges::readEdge(istream &input, const BGSIZE iEdg) {
   int synapse_type(0);

   // input.ignore() so input skips over end-of-line characters.
   input >> sourceNeuronIndex_[iEdg];
   input.ignore();
   input >> destNeuronIndex_[iEdg];
   input.ignore();
   input >> W_[iEdg];
   input.ignore();
   input >> psr_[iEdg];
   input.ignore();
   input >> synapse_type;
   input.ignore();
   input >> inUse_[iEdg];
   input.ignore();

   type_[iEdg] = synapseOrdinalToType(synapse_type);
}

///  Write the synapse data to the stream.
///
///  @param  output  stream to print out to.
///  @param  iEdg    Index of the synapse to print out.
void AllEdges::writeEdge(ostream &output, const BGSIZE iEdg) const {
   output << sourceNeuronIndex_[iEdg] << ends;
   output << destNeuronIndex_[iEdg] << ends;
   output << W_[iEdg] << ends;
   output << psr_[iEdg] << ends;
   output << type_[iEdg] << ends;
   output << inUse_[iEdg] << ends;
}

///  Create a synapse index map.
EdgeIndexMap *AllEdges::createEdgeIndexMap() {
   int neuronCount = Simulator::getInstance().getTotalVertices();
   int totalSynapseCount = 0;

   // count the total edges
   for (int i = 0; i < neuronCount; i++) {
      assert(static_cast<int>(synapseCounts_[i]) < Simulator::getInstance().getMaxEdgesPerVertex());
      totalSynapseCount += synapseCounts_[i];
   }

   DEBUG (cout << "totalSynapseCount: " << totalSynapseCount << endl;)

   if (totalSynapseCount == 0) {
      return nullptr;
   }

   // allocate memories for forward map
   vector<BGSIZE> *rgSynapseSynapseIndexMap = new vector<BGSIZE>[neuronCount];

   BGSIZE syn_i = 0;
   int numInUse = 0;

   // create synapse forward map & active synapse map
   EdgeIndexMap *edgeIndexMap = new EdgeIndexMap(neuronCount, totalSynapseCount);
   for (int i = 0; i < neuronCount; i++) {
      BGSIZE edge_count = 0;
      edgeIndexMap->incomingSynapseBegin_[i] = numInUse;
      for (int j = 0; j < Simulator::getInstance().getMaxEdgesPerVertex(); j++, syn_i++) {
         if (inUse_[syn_i] == true) {
            int idx = sourceNeuronIndex_[syn_i];
            rgSynapseSynapseIndexMap[idx].push_back(syn_i);

            edgeIndexMap->incomingSynapseIndexMap_[numInUse] = syn_i;
            numInUse++;
            edge_count++;
         }
      }
      assert(edge_count == this->synapseCounts_[i]);
      edgeIndexMap->incomingSynapseCount_[i] = edge_count;
   }

   assert(totalSynapseCount == numInUse);
   this->totalEdgeCount_ = totalSynapseCount;

   syn_i = 0;
   for (int i = 0; i < neuronCount; i++) {
      edgeIndexMap->outgoingSynapseBegin_[i] = syn_i;
      edgeIndexMap->outgoingSynapseCount_[i] = rgSynapseSynapseIndexMap[i].size();

      for (BGSIZE j = 0; j < rgSynapseSynapseIndexMap[i].size(); j++, syn_i++) {
         edgeIndexMap->outgoingSynapseIndexMap_[syn_i] = rgSynapseSynapseIndexMap[i][j];
      }
   }

   // delete memories
   delete[] rgSynapseSynapseIndexMap;

   return edgeIndexMap;
}
   
///  Returns an appropriate synapseType object for the given integer.
///
///  @param  typeOrdinal    Integer that correspond with a synapseType.
///  @return the SynapseType that corresponds with the given integer.
synapseType AllEdges::synapseOrdinalToType(const int typeOrdinal) {
   switch (typeOrdinal) {
      case 0:
         return II;
      case 1:
         return IE;
      case 2:
         return EI;
      case 3:
         return EE;
      default:
         return STYPE_UNDEF;
   }
}

#if !defined(USE_GPU)

///  Advance all the Synapses in the simulation.
///
///  @param  vertices           The vertex list to search from.
///  @param  edgeIndexMap   Pointer to EdgeIndexMap structure.
void AllEdges::advanceEdges(IAllVertices *vertices, EdgeIndexMap *edgeIndexMap) {
   for (BGSIZE i = 0; i < totalEdgeCount_; i++) {
      BGSIZE iEdg = edgeIndexMap->incomingSynapseIndexMap_[i];
      advanceEdge(iEdg, vertices);
   }
}

///  Remove a synapse from the network.
///
///  @param  neuronIndex    Index of a vertex to remove from.
///  @param  iEdg           Index of a synapse to remove.
void AllEdges::eraseEdge(const int neuronIndex, const BGSIZE iEdg) {
   synapseCounts_[neuronIndex]--;
   inUse_[iEdg] = false;
   summationPoint_[iEdg] = nullptr;
   W_[iEdg] = 0;
}

#endif // !defined(USE_GPU)

///  Adds a Synapse to the model, connecting two Neurons.
///
///  @param  iEdg        Index of the synapse to be added.
///  @param  type        The type of the Synapse to add.
///  @param  srcVertex  The Neuron that sends to this Synapse.
///  @param  destVertex The Neuron that receives from the Synapse.
///  @param  sumPoint   Summation point address.
///  @param  deltaT      Inner simulation step duration
void
AllEdges::addEdge(BGSIZE &iEdg, synapseType type, const int srcVertex, const int destVertex, BGFLOAT *sumPoint,
                        const BGFLOAT deltaT) {
   if (synapseCounts_[destVertex] >= maxEdgesPerVertex_) {
      LOG4CPLUS_FATAL(fileLogger_, "Neuron : " << destVertex << " ran out of space for new edges.");
      throw runtime_error("Neuron : " + destVertex + string(" ran out of space for new edges."));
   }

   // add it to the list
   BGSIZE synapseIndex;
   for (synapseIndex = 0; synapseIndex < maxEdgesPerVertex_; synapseIndex++) {
      iEdg = maxEdgesPerVertex_ * destVertex + synapseIndex;
      if (!inUse_[iEdg]) {
         break;
      }
   }

   synapseCounts_[destVertex]++;

   // create a synapse
   createEdge(iEdg, srcVertex, destVertex, sumPoint, deltaT, type);
}

///  Get the sign of the synapseType.
///
///  @param    type    synapseType I to I, I to E, E to I, or E to E
///  @return   1 or -1, or 0 if error
int AllEdges::edgSign(const synapseType type) {
   switch (type) {
      case II:
      case IE:
         return -1;
      case EI:
      case EE:
         return 1;
      case STYPE_UNDEF:
         // TODO error.
         return 0;
   }

   return 0;
}

///  Prints SynapsesProps data to console.
void AllEdges::printSynapsesProps() const {
   cout << "This is SynapsesProps data:" << endl;
   for (int i = 0; i < maxEdgesPerVertex_ * countVertices_; i++) {
      if (W_[i] != 0.0) {
         cout << "W[" << i << "] = " << W_[i];
         cout << " sourNeuron: " << sourceNeuronIndex_[i];
         cout << " desNeuron: " << destNeuronIndex_[i];
         cout << " type: " << type_[i];
         cout << " psr: " << psr_[i];
         cout << " in_use:" << inUse_[i];
         if (summationPoint_[i] != nullptr) {
            cout << " summationPoint: is created!" << endl;
         } else {
            cout << " summationPoint: is EMPTY!!!!!" << endl;
         }
      }
   }

   for (int i = 0; i < countVertices_; i++) {
      cout << "edge_counts:" << "vertex[" << i << "]" << synapseCounts_[i] << endl;
   }

   cout << "totalSynapseCount:" << totalEdgeCount_ << endl;
   cout << "maxEdgesPerVertex:" << maxEdgesPerVertex_ << endl;
   cout << "count_neurons:" << countVertices_ << endl;
}





