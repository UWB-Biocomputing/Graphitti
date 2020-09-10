/* ------------- CONNECTIONS STRUCT ------------ *\
 * Below all of the resources for the various
 * connections are instantiated and initialized.
 * All of the allocation for memory is done in the
 * constructor’s parameters and not in the body of
 * the function. Once all memory has been allocated
 * the constructor fills in known information
 * into “radii” and “rates”.
\* --------------------------------------------- */
// TODO comment
/* ------------------- ERROR ------------------- *\
 * terminate called after throwing an instance of 'std::bad_alloc'
 *      what():  St9bad_alloc
 * ------------------- CAUSE ------------------- *|
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
 * ----------------- 1/25/14 ------------------- *|
 * Currently when running a simulation of sizes
 * equal to or greater than 100 * 100 the above
 * error is thrown. After some testing we have
 * determined that this is a hardware dependent
 * issue, not software. We are also looking into
 * switching matrix types from "complete" to
 * "sparce". If successful it is possible the
 * problematic matricies mentioned above will use
 * only 1/250 of their current space.
\* --------------------------------------------- */

#include "ConnGrowth.h"
#include "ParseParamError.h"
#include "IAllSynapses.h"
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
   W_ = NULL;
   radii_ = NULL;
   rates_ = NULL;
   delta_ = NULL;
   area_ = NULL;
   outgrowth_ = NULL;
   deltaR_ = NULL;
   radiiSize_ = 0;
}

ConnGrowth::~ConnGrowth() {
   cleanupConnections();
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  layout    Layout information of the neunal network.
 *  @param  neurons   The Neuron list to search from.
 *  @param  synapses  The Synapse list to search from.
 */
void ConnGrowth::setupConnections(Layout *layout, IAllNeurons *neurons, IAllSynapses *synapses) {
   int num_neurons = Simulator::getInstance().getTotalNeurons();
   radiiSize_ = num_neurons;

   W_ = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons, 0);
   radii_ = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons, growthParams_.startRadius);
   rates_ = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons, 0);
   delta_ = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons);
   area_ = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons, 0);
   outgrowth_ = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons);
   deltaR_ = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons);

   // Init connection frontier distance change matrix with the current distances
   (*delta_) = (*layout->dist_);
}

/*
 *  Cleanup the class (deallocate memories).
 */
void ConnGrowth::cleanupConnections() {
   if (W_ != NULL) delete W_;
   if (radii_ != NULL) delete radii_;
   if (rates_ != NULL) delete rates_;
   if (delta_ != NULL) delete delta_;
   if (area_ != NULL) delete area_;
   if (outgrowth_ != NULL) delete outgrowth_;
   if (deltaR_ != NULL) delete deltaR_;

   W_ = NULL;
   radii_ = NULL;
   rates_ = NULL;
   delta_ = NULL;
   area_ = NULL;
   outgrowth_ = NULL;
   deltaR_ = NULL;
   radiiSize_ = 0;
}

/**
 * Load member variables from configuration file.
 * Registered to OperationManager as Operations::op::loadParameters
 */
void ConnGrowth::loadParameters() {
   ParameterManager::getInstance().getBGFloatByXpath("//GrowthParams/epsilon/text()", growthParams_.epsilon);
   ParameterManager::getInstance().getBGFloatByXpath("//GrowthParams/beta/text()", growthParams_.beta);
   ParameterManager::getInstance().getBGFloatByXpath("//GrowthParams/rho/text()", growthParams_.rho);
   ParameterManager::getInstance().getBGFloatByXpath("//GrowthParams/targetRate/text()", growthParams_.targetRate);
   ParameterManager::getInstance().getBGFloatByXpath("//GrowthParams/minRadius/text()", growthParams_.minRadius);
   ParameterManager::getInstance().getBGFloatByXpath("//GrowthParams/startRadius/text()", growthParams_.startRadius);
}

/*
 *  Prints out all parameters of the connections to ostream.
 *
 *  @param  output  ostream to send output to.
 */
void ConnGrowth::printParameters() const {
   cout << "CONNECTIONS PARAMETERS" << endl;
   cout << "\tConnections type: ConnGrowth" << endl;
   cout << "\tepsilon: " << growthParams_.epsilon << endl;
   cout << "\tbeta: " << growthParams_.beta << endl;
   cout << "\trho: " << growthParams_.rho << endl;
   cout << "\tTarget rate: " << growthParams_.targetRate << "," << endl;
   cout << "\tMinimum radius: " << growthParams_.minRadius << endl;
   cout << "\tStarting raduis: " << growthParams_.startRadius << endl << endl;
}

/*
 *  Update the connections status in every epoch.
 *
 *  @param  neurons  The Neuron list to search from.
 *  @param  layout   Layout information of the neunal network.
 *  @return true if successful, false otherwise.
 */
bool ConnGrowth::updateConnections(IAllNeurons &neurons, Layout *layout) {
   // Update Connections data
   updateConns(neurons);

   // Update the distance between frontiers of Neurons
   updateFrontiers(Simulator::getInstance().getTotalNeurons(), layout);

   // Update the areas of overlap in between Neurons
   updateOverlap(Simulator::getInstance().getTotalNeurons(), layout);

   return true;
}

/*
 *  Calculates firing rates, neuron radii change and assign new values.
 *
 *  @param  neurons  The Neuron list to search from.
 */
void ConnGrowth::updateConns(IAllNeurons &neurons) {
   AllSpikingNeurons &spNeurons = dynamic_cast<AllSpikingNeurons &>(neurons);

   // Calculate growth cycle firing rate for previous period
   int max_spikes = static_cast<int> (Simulator::getInstance().getEpochDuration() *
                                      Simulator::getInstance().getMaxFiringRate());
   for (int i = 0; i < Simulator::getInstance().getTotalNeurons(); i++) {
      // Calculate firing rate
      assert(spNeurons.spikeCount_[i] < max_spikes);
      (*rates_)[i] = spNeurons.spikeCount_[i] / Simulator::getInstance().getEpochDuration();
   }

   // compute neuron radii change and assign new values
   (*outgrowth_) =
         1.0 - 2.0 / (1.0 + exp((growthParams_.epsilon - *rates_ / growthParams_.maxRate) / growthParams_.beta));
   (*deltaR_) = Simulator::getInstance().getEpochDuration() * growthParams_.rho * *outgrowth_;
   (*radii_) += (*deltaR_);
}

/*
 *  Update the distance between frontiers of Neurons.
 *
 *  @param  num_neurons Number of neurons to update.
 *  @param  layout      Layout information of the neunal network.
 */
void ConnGrowth::updateFrontiers(const int num_neurons, Layout *layout) {
   DEBUG(cout << "Updating distance between frontiers..." << endl;)
   // Update distance between frontiers
   for (int unit = 0; unit < num_neurons - 1; unit++) {
      for (int i = unit + 1; i < num_neurons; i++) {
         (*delta_)(unit, i) = (*layout->dist_)(unit, i) - ((*radii_)[unit] + (*radii_)[i]);
         (*delta_)(i, unit) = (*delta_)(unit, i);
      }
   }
}

/*
 *  Update the areas of overlap in between Neurons.
 *
 *  @param  num_neurons Number of Neurons to update.
 *  @param  layout      Layout information of the neunal network.
 */
void ConnGrowth::updateOverlap(BGFLOAT num_neurons, Layout *layout) {
   DEBUG(cout << "computing areas of overlap" << endl;)

   // Compute areas of overlap; this is only done for overlapping units
   for (int i = 0; i < num_neurons; i++) {
      for (int j = 0; j < num_neurons; j++) {
         (*area_)(i, j) = 0.0;

         if ((*delta_)(i, j) < 0) {
            BGFLOAT lenAB = (*layout->dist_)(i, j);
            BGFLOAT r1 = (*radii_)[i];
            BGFLOAT r2 = (*radii_)[j];

            if (lenAB + min(r1, r2) <= max(r1, r2)) {
               (*area_)(i, j) = pi * min(r1, r2) * min(r1, r2); // Completely overlapping unit

#ifdef LOGFILE
               logFile << "Completely overlapping (i, j, r1, r2, area): "
                   << i << ", " << j << ", " << r1 << ", " << r2 << ", " << *pAarea(i, j) << endl;
#endif // LOGFILE
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

/*
 *  Update the weight of the Synapses in the simulation.
 *  To be clear, iterates through all source and destination neurons
 *  and updates their synaptic strengths from the weight matrix.
 *  Note: Platform Dependent.
 *
 *  @param  num_neurons Number of neurons to update.
 *  @param  ineurons    the AllNeurons object.
 *  @param  isynapses   the AllSynapses object.
 *  @param  layout      the Layout object.
 */
void ConnGrowth::updateSynapsesWeights(const int num_neurons, IAllNeurons &ineurons, IAllSynapses &isynapses,
                                       Layout *layout) {
   AllNeurons &neurons = dynamic_cast<AllNeurons &>(ineurons);
   AllSynapses &synapses = dynamic_cast<AllSynapses &>(isynapses);

   // For now, we just set the weights to equal the areas. We will later
   // scale it and set its sign (when we index and get its sign).
   (*W_) = (*area_);

   int adjusted = 0;
   int could_have_been_removed = 0; // TODO: use this value
   int removed = 0;
   int added = 0;

   DEBUG(cout << "adjusting weights" << endl;)

   // Scale and add sign to the areas
   // visit each neuron 'a'
   for (int src_neuron = 0; src_neuron < num_neurons; src_neuron++) {
      // and each destination neuron 'b'
      for (int dest_neuron = 0; dest_neuron < num_neurons; dest_neuron++) {
         // visit each synapse at (xa,ya)
         bool connected = false;
         synapseType type = layout->synType(src_neuron, dest_neuron);

         // for each existing synapse
         BGSIZE synapse_counts = synapses.synapseCounts_[dest_neuron];
         BGSIZE synapse_adjusted = 0;
         BGSIZE iSyn = Simulator::getInstance().getMaxSynapsesPerNeuron() * dest_neuron;
         for (BGSIZE synapse_index = 0; synapse_adjusted < synapse_counts; synapse_index++, iSyn++) {
            if (synapses.inUse_[iSyn] == true) {
               // if there is a synapse between a and b
               if (synapses.sourceNeuronIndex_[iSyn] == src_neuron) {
                  connected = true;
                  adjusted++;
                  // adjust the strength of the synapse or remove
                  // it from the synapse map if it has gone below
                  // zero.
                  if ((*W_)(src_neuron, dest_neuron) <= 0) {
                     removed++;
                     synapses.eraseSynapse(dest_neuron, iSyn);
                  } else {
                     // adjust
                     // SYNAPSE_STRENGTH_ADJUSTMENT is 1.0e-8;
                     synapses.W_[iSyn] = (*W_)(src_neuron, dest_neuron) *
                                         synapses.synSign(type) * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;

                     DEBUG_MID(cout << "weight of rgSynapseMap" <<
                                    "[" << synapse_index << "]: " <<
                                    synapses.W_[iSyn] << endl;);
                  }
               }
               synapse_adjusted++;

            }
         }

         // if not connected and weight(a,b) > 0, add a new synapse from a to b
         if (!connected && ((*W_)(src_neuron, dest_neuron) > 0)) {

            // locate summation point
            BGFLOAT *sum_point = &(neurons.summationMap_[dest_neuron]);
            added++;

            BGSIZE iSyn;
            synapses.addSynapse(iSyn, type, src_neuron, dest_neuron, sum_point,
                                Simulator::getInstance().getDeltaT());
            synapses.W_[iSyn] =
                  (*W_)(src_neuron, dest_neuron) * synapses.synSign(type) *
                  AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;

         }
      }
   }

   DEBUG (cout << "adjusted: " << adjusted << endl;)
   DEBUG (cout << "could have been removed (TODO: calculate this): " << could_have_been_removed << endl;)
   DEBUG (cout << "removed: " << removed << endl;)
   DEBUG (cout << "added: " << added << endl << endl << endl;)
}

#endif // !USE_GPU

/*
 *  Creates a recorder class object for the connection.
 *  This function tries to create either Xml recorder or
 *  Hdf5 recorder based on the extension of the file name.
 *
 *  @return Pointer to the recorder class object.
 */
IRecorder *ConnGrowth::createRecorder() {
   // create & init simulation recorder
   IRecorder *simRecorder = NULL;
   if (Simulator::getInstance().getResultFileName().find(".xml") != string::npos) {
      simRecorder = new XmlRecorder();
   }
#ifdef USE_HDF5
      else if (simInfo->resultFileName.find(".h5") != string::npos) {
          simRecorder = new Hdf5GrowthRecorder(simInfo);
      }
#endif // USE_HDF5
   else {
      return NULL;
   }
   if (simRecorder != NULL) {
      simRecorder->init();
   }

   return simRecorder;
}

/**
 *  Prints radii 
 */
void ConnGrowth::printRadii() const {
   for (int i = 0; i < radiiSize_; i++) {
      cout << "radii[" << i << "] = " << (*radii_)[i] << endl;
   }
}


