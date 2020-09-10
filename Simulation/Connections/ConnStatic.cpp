#include "ConnStatic.h"
#include "ParseParamError.h"
#include "IAllSynapses.h"
#include "AllNeurons.h"
#include "AllSynapses.h"
#include "OperationManager.h"

#include "XmlRecorder.h"

#ifdef USE_HDF5
#include "Hdf5Recorder.h"
#endif

#include <algorithm>

ConnStatic::ConnStatic() {
   threshConnsRadius_ = 0;
   connsPerNeuron_ = 0;
   rewiringProbability_ = 0;
}

ConnStatic::~ConnStatic() {
   cleanupConnections();
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *  Initialize the small world network characterized by parameters: 
 *  number of maximum connections per neurons, connection radius threshold, and
 *  small-world rewiring probability.
 *
 *  @param  layout    Layout information of the neunal network.
 *  @param  neurons   The Neuron list to search from.
 *  @param  synapses  The Synapse list to search from.
 */
void ConnStatic::setupConnections(Layout *layout, IAllNeurons *neurons, IAllSynapses *synapses) {
   int num_neurons = Simulator::getInstance().getTotalNeurons();
   vector<DistDestNeuron> distDestNeurons[num_neurons];

   int added = 0;

   DEBUG(cout << "Initializing connections" << endl;)

   for (int src_neuron = 0; src_neuron < num_neurons; src_neuron++) {
      distDestNeurons[src_neuron].clear();

      // pick the connections shorter than threshConnsRadius
      for (int dest_neuron = 0; dest_neuron < num_neurons; dest_neuron++) {
         if (src_neuron != dest_neuron) {
            BGFLOAT dist = (*layout->dist_)(src_neuron, dest_neuron);
            if (dist <= threshConnsRadius_) {
               DistDestNeuron distDestNeuron;
               distDestNeuron.dist = dist;
               distDestNeuron.dest_neuron = dest_neuron;
               distDestNeurons[src_neuron].push_back(distDestNeuron);
            }
         }
      }

      // sort ascendant
      sort(distDestNeurons[src_neuron].begin(), distDestNeurons[src_neuron].end());
      // pick the shortest m_nConnsPerNeuron connections
      for (BGSIZE i = 0; i < distDestNeurons[src_neuron].size() && (int) i < connsPerNeuron_; i++) {
         int dest_neuron = distDestNeurons[src_neuron][i].dest_neuron;
         synapseType type = layout->synType(src_neuron, dest_neuron);
         BGFLOAT *sum_point = &(dynamic_cast<AllNeurons *>(neurons)->summationMap_[dest_neuron]);

         DEBUG_MID (cout << "source: " << src_neuron << " dest: " << dest_neuron << " dist: "
                         << distDestNeurons[src_neuron][i].dist << endl;)

         BGSIZE iSyn;
         synapses->addSynapse(iSyn, type, src_neuron, dest_neuron, sum_point, Simulator::getInstance().getDeltaT());
         added++;

         // set synapse weight
         // TODO: we need another synaptic weight distibution mode (normal distribution)
         if (synapses->synSign(type) > 0) {
            dynamic_cast<AllSynapses *>(synapses)->W_[iSyn] = rng.inRange(excWeight_[0], excWeight_[1]);
         } else {
            dynamic_cast<AllSynapses *>(synapses)->W_[iSyn] = rng.inRange(inhWeight_[0], inhWeight_[1]);
         }
      }
   }

   int nRewiring = added * rewiringProbability_;

   DEBUG(cout << "Rewiring connections: " << nRewiring << endl;)

   DEBUG (cout << "added connections: " << added << endl << endl << endl;)
}

/*
 *  Cleanup the class.
 */
void ConnStatic::cleanupConnections() {
}

/*
 * Load member variables from configuration file.
 * Registered to OperationManager as Operations::op::loadParameters
 */
void ConnStatic::loadParameters() {
   // ConnStatic doesn't have any parameters to load from the configuration file.
}


/*
 *  Prints out all parameters of the connections to console.
 */
void ConnStatic::printParameters() const {
   cout << "CONNECTIONS PARAMETERS" << endl;
   cout << "\tConnections Type: ConnStatic" << endl;
   cout << "\tConnection radius threshold: " << threshConnsRadius_ << endl;
   cout << "\tConnections per neuron: " << connsPerNeuron_ << endl;
   cout << "\tRewiring probability: " << rewiringProbability_ << endl << endl;
}

/*
 *  Creates a recorder class object for the connection.
 *  This function tries to create either Xml recorder or
 *  Hdf5 recorder based on the extension of the file name.
 *
 *  @return Pointer to the recorder class object.
 */
IRecorder *ConnStatic::createRecorder() {
   // create & init simulation recorder
   IRecorder *simRecorder = NULL;
   if (Simulator::getInstance().getResultFileName().find(".xml") != string::npos) {
      simRecorder = new XmlRecorder();
   }
#ifdef USE_HDF5
      else if (Simulator::getInstance().getStateOutputFileName().find(".h5") != string::npos) {
          simRecorder = new Hdf5Recorder();
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

