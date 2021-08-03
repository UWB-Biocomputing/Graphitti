/**
 * @file ConnGrowth.cpp
 * 
 * @ingroup Simulator/Connections
 * 
 * @brief The model of the activity dependent neurite outgrowth
 *
 * Below all of the resources for the various
 * connections are instantiated and initialized.
 * All of the allocation for memory is done in the
 * constructor’s parameters and not in the body of
 * the function. Once all memory has been allocated
 * the constructor fills in known information
 * into “radii” and “rates”.
 * 
 * ERROR
 * terminate called after throwing an instance of 'std::bad_alloc'
 *      what():  St9bad_alloc
 * CAUSE
 * As simulations expand in size the number of
 * neurons in total increases exponentially. When
 * using a MATRIX_TYPE = “complete” the amount of
 * used memory increases by another order of magnitude.
 * Once enough memory is used no more memory can be
 * allocated and a “bsd_alloc” will be thrown.
 * The following members of the connection constructor
 * consume equally vast amounts of memory as the
 * simulation sizes grow:
 *      - W             - radii
 *      - rates         - dist2
 *      - delta         - dist
 *      - areai
 */

#include "ConnGrowth.h"
#include "ParseParamError.h"
#include "AllEdges.h"
#include "XmlGrowthRecorder.h"
#include "AllSpikingNeurons.h"
#include "Matrix/CompleteMatrix.h"
#include "Matrix/Matrix.h"
#include "Matrix/VectorMatrix.h"
#include "ParameterManager.h"
#include "OperationManager.h"

#ifdef USE_HDF5
#include "Hdf5GrowthRecorder.h"
#endif

ConnGrowth::ConnGrowth() : Connections() {
   W_ = nullptr;
   radii_ = nullptr;
   rates_ = nullptr;
   delta_ = nullptr;
   area_ = nullptr;
   outgrowth_ = nullptr;
   deltaR_ = nullptr;
   radiiSize_ = 0;
}

ConnGrowth::~ConnGrowth() {
   if (W_ != nullptr) delete W_;
   if (radii_ != nullptr) delete radii_;
   if (rates_ != nullptr) delete rates_;
   if (delta_ != nullptr) delete delta_;
   if (area_ != nullptr) delete area_;
   if (outgrowth_ != nullptr) delete outgrowth_;
   if (deltaR_ != nullptr) delete deltaR_;

   W_ = nullptr;
   radii_ = nullptr;
   rates_ = nullptr;
   delta_ = nullptr;
   area_ = nullptr;
   outgrowth_ = nullptr;
   deltaR_ = nullptr;
   radiiSize_ = 0;
}

///  Setup the internal structure of the class (allocate memories and initialize them).
///
///  @param  layout    Layout information of the neural network.
///  @param  vertices   The vertex list to search from.
///  @param  synapses  The Synapse list to search from.
void ConnGrowth::setupConnections(Layout *layout, AllVertices *vertices, AllEdges *synapses) {
   int numVertices = Simulator::getInstance().getTotalVertices();
   radiiSize_ = numVertices;

   W_ = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, numVertices, numVertices, 0);
   radii_ = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, numVertices, growthParams_.startRadius);
   rates_ = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, numVertices, 0);
   delta_ = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, numVertices, numVertices);
   area_ = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, numVertices, numVertices, 0);
   outgrowth_ = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, numVertices);
   deltaR_ = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, numVertices);

   // Init connection frontier distance change matrix with the current distances
   (*delta_) = (*layout->dist_);
}

/// Load member variables from configuration file.
/// Registered to OperationManager as Operations::op::loadParameters
void ConnGrowth::loadParameters() {
   ParameterManager::getInstance().getBGFloatByXpath("//GrowthParams/epsilon/text()", growthParams_.epsilon);
   ParameterManager::getInstance().getBGFloatByXpath("//GrowthParams/beta/text()", growthParams_.beta);
   ParameterManager::getInstance().getBGFloatByXpath("//GrowthParams/rho/text()", growthParams_.rho);
   ParameterManager::getInstance().getBGFloatByXpath("//GrowthParams/targetRate/text()", growthParams_.targetRate);
   ParameterManager::getInstance().getBGFloatByXpath("//GrowthParams/minRadius/text()", growthParams_.minRadius);
   ParameterManager::getInstance().getBGFloatByXpath("//GrowthParams/startRadius/text()", growthParams_.startRadius);

   // initial maximum firing rate
   if (growthParams_.epsilon != 0) {
      growthParams_.maxRate = growthParams_.targetRate / growthParams_.epsilon;
	} else {
      LOG4CPLUS_FATAL(fileLogger_, "Parameter GrowthParams/epsilon/ has a value of 0" << endl);
      exit(EXIT_FAILURE);
   }
}

/// Prints out all parameters to logging file.
/// Registered to OperationManager as Operation::printParameters
void ConnGrowth::printParameters() const {
   LOG4CPLUS_DEBUG(fileLogger_, "\nCONNECTIONS PARAMETERS" << endl
    << "\tConnections type: ConnGrowth" << endl
    << "\tepsilon: " << growthParams_.epsilon << endl
    << "\tbeta: " << growthParams_.beta << endl
    << "\trho: " << growthParams_.rho << endl
    << "\tTarget rate: " << growthParams_.targetRate << "," << endl
    << "\tMinimum radius: " << growthParams_.minRadius << endl
    << "\tStarting radius: " << growthParams_.startRadius << endl << endl);
}

///  Update the connections status in every epoch.
///
///  @param  vertices  The vertex list to search from.
///  @param  layout   Layout information of the neural network.
///  @return true if successful, false otherwise.
bool ConnGrowth::updateConnections(AllVertices &vertices, Layout *layout) {
   // Update Connections data
   updateConns(vertices);

   // Update the distance between frontiers of vertices
   updateFrontiers(Simulator::getInstance().getTotalVertices(), layout);

   // Update the areas of overlap in between vertices
   updateOverlap(Simulator::getInstance().getTotalVertices(), layout);

   return true;
}

///  Calculates firing rates, vertex radii change and assign new values.
///
///  @param  vertices  The vertex list to search from.
void ConnGrowth::updateConns(AllVertices &vertices) {
   AllSpikingNeurons &spNeurons = dynamic_cast<AllSpikingNeurons &>(vertices);

   // Calculate growth cycle firing rate for previous period
   int maxSpikes = static_cast<int> (Simulator::getInstance().getEpochDuration() *
                                      Simulator::getInstance().getMaxFiringRate());
   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      // Calculate firing rate
      assert(spNeurons.spikeCount_[i] < maxSpikes);
      (*rates_)[i] = spNeurons.spikeCount_[i] / Simulator::getInstance().getEpochDuration();
   }
   
   // compute vertex radii change and assign new values
   (*outgrowth_) =
         1.0 - 2.0 / (1.0 + exp((growthParams_.epsilon - *rates_ / growthParams_.maxRate) / growthParams_.beta));
   (*deltaR_) = Simulator::getInstance().getEpochDuration() * growthParams_.rho * *outgrowth_;
   (*radii_) += (*deltaR_);
}

///  Update the distance between frontiers of vertices.
///
///  @param  numVertices  Number of vertices to update.
///  @param  layout      Layout information of the neural network.
void ConnGrowth::updateFrontiers(const int numVertices, Layout *layout) {
   LOG4CPLUS_INFO(fileLogger_, "Updating distance between frontiers...");
   // Update distance between frontiers
   for (int unit = 0; unit < numVertices - 1; unit++) {
      for (int i = unit + 1; i < numVertices; i++) {
         (*delta_)(unit, i) = (*layout->dist_)(unit, i) - ((*radii_)[unit] + (*radii_)[i]);
         (*delta_)(i, unit) = (*delta_)(unit, i);
      }
   }
}

///  Update the areas of overlap in between Neurons.
///
///  @param  numVertices  Number of vertices to update.
///  @param  layout      Layout information of the neural network.
void ConnGrowth::updateOverlap(BGFLOAT numVertices, Layout *layout) {
   LOG4CPLUS_INFO(fileLogger_, "Computing areas of overlap");

   // Compute areas of overlap; this is only done for overlapping units
   for (int i = 0; i < numVertices; i++) {
      for (int j = 0; j < numVertices; j++) {
         (*area_)(i, j) = 0.0;

         if ((*delta_)(i, j) < 0) {
            BGFLOAT lenAB = (*layout->dist_)(i, j);
            BGFLOAT r1 = (*radii_)[i];
            BGFLOAT r2 = (*radii_)[j];

            if (lenAB + min(r1, r2) <= max(r1, r2)) {
               (*area_)(i, j) = pi * min(r1, r2) * min(r1, r2); // Completely overlapping unit

               LOG4CPLUS_DEBUG(fileLogger_, "Completely overlapping (i, j, r1, r2, area): "
                     << i << ", " << j << ", " << r1 << ", " << r2 << ", " << *area_ << endl);
            } else {
               // Partially overlapping unit
               BGFLOAT lenAB2 = (*layout->dist2_)(i, j);
               BGFLOAT r12 = r1 * r1;
               BGFLOAT r22 = r2 * r2;

               BGFLOAT cosCBA = (r22 + lenAB2 - r12) / (2.0 * r2 * lenAB);
               BGFLOAT cosCAB = (r12 + lenAB2 - r22) / (2.0 * r1 * lenAB);

               if (fabs(cosCBA) >= 1.0 || fabs(cosCAB) >= 1.0) {
                  (*area_)(i, j) = 0.0;
               } else {

                  BGFLOAT angCBA = acos(cosCBA);
                  BGFLOAT angCBD = 2.0 * angCBA;

                  BGFLOAT angCAB = acos(cosCAB);
                  BGFLOAT angCAD = 2.0 * angCAB;

                  (*area_)(i, j) = 0.5 * (r22 * (angCBD - sin(angCBD)) + r12 * (angCAD - sin(angCAD)));
               }
            }
         }
      }
   }
}

#if !defined(USE_GPU)

///  Update the weight of the Synapses in the simulation.
///  To be clear, iterates through all source and destination neurons
///  and updates their synaptic strengths from the weight matrix.
///  Note: Platform Dependent.
///
///  @param  numVertices  Number of vertices to update.
///  @param  ivertices    the AllVertices object.
///  @param  iedges   the AllEdges object.
///  @param  layout      the Layout object.
void ConnGrowth::updateSynapsesWeights(const int numVertices, AllVertices &vertices, AllEdges &iedges,
                                       Layout *layout) {
   AllNeuroEdges &synapses = dynamic_cast<AllNeuroEdges &>(iedges);

   // For now, we just set the weights to equal the areas. We will later
   // scale it and set its sign (when we index and get its sign).
   (*W_) = (*area_);

   int adjusted = 0;
   int couldBeRemoved = 0; // TODO: use this value
   int removed = 0;
   int added = 0;

   LOG4CPLUS_INFO(fileLogger_, "Adjusting Synapse weights");

   // Scale and add sign to the areas
   // visit each neuron 'a'
   for (int srcVertex = 0; srcVertex < numVertices; srcVertex++) {
      // and each destination neuron 'b'
      for (int destVertex = 0; destVertex < numVertices; destVertex++) {
         // visit each synapse at (xa,ya)
         bool connected = false;
         edgeType type = layout->edgType(srcVertex, destVertex);

         // for each existing synapse
         BGSIZE synapseCounts = synapses.edgeCounts_[destVertex];
         BGSIZE synapse_adjusted = 0;
         BGSIZE iEdg = Simulator::getInstance().getMaxEdgesPerVertex() * destVertex;
         for (BGSIZE synapseIndex = 0; synapse_adjusted < synapseCounts; synapseIndex++, iEdg++) {
            if (synapses.inUse_[iEdg] == true) {
               // if there is a synapse between a and b
               if (synapses.sourceVertexIndex_[iEdg] == srcVertex) {
                  connected = true;
                  adjusted++;
                  // adjust the strength of the synapse or remove
                  // it from the synapse map if it has gone below
                  // zero.
                  if ((*W_)(srcVertex, destVertex) <= 0) {
                     removed++;
                     synapses.eraseEdge(destVertex, iEdg);
                  } else {
                     // adjust
                     // SYNAPSE_STRENGTH_ADJUSTMENT is 1.0e-8;
                     synapses.W_[iEdg] = (*W_)(srcVertex, destVertex) *
                                         synapses.edgSign(type) * AllNeuroEdges::SYNAPSE_STRENGTH_ADJUSTMENT;

                     LOG4CPLUS_DEBUG(fileLogger_, "Weight of rgSynapseMap" <<
                                                                           "[" << synapseIndex << "]: " <<
                                                                           synapses.W_[iEdg]);
                  }
               }
               synapse_adjusted++;

            }
         }

         // if not connected and weight(a,b) > 0, add a new synapse from a to b
         if (!connected && ((*W_)(srcVertex, destVertex) > 0)) {

            // locate summation point
            BGFLOAT *sumPoint = &(vertices.summationMap_[destVertex]);
            added++;

            BGSIZE iEdg;
            synapses.addEdge(iEdg, type, srcVertex, destVertex, sumPoint,
                                Simulator::getInstance().getDeltaT());
            synapses.W_[iEdg] =
                  (*W_)(srcVertex, destVertex) * synapses.edgSign(type) *
                  AllNeuroEdges::SYNAPSE_STRENGTH_ADJUSTMENT;

         }
      }
   }

   LOG4CPLUS_INFO(fileLogger_, "\nAdjusted: " << adjusted
             << "\nCould have been removed (TODO: calculate this): " << couldBeRemoved
             << "\nRemoved: " << removed
             << "\nAdded: " << added);
}

#endif // !USE_GPU

///  Prints radii 
void ConnGrowth::printRadii() const {
   for (int i = 0; i < radiiSize_; i++) {
      cout << "radii[" << i << "] = " << (*radii_)[i] << endl;
   }
}


