#include "ConnStatic.h"
#include "ParseParamError.h"
#include "IAllSynapses.h"
#include "AllNeurons.h"
#include "AllSynapses.h"
#include "OperationManager.h"
#include "ParameterManager.h"
#include "XmlRecorder.h"

#ifdef USE_HDF5
#include "Hdf5Recorder.h"
#endif

#include <algorithm>

ConnStatic::ConnStatic() {
    threshConnsRadius_ = 4;
   connsPerNeuron_ = 2;
   rewiringProbability_ = 0.75;
   WSTDP_=NULL;
   //excWeight_[0]=0;
   //excWeight_[1]=2.5e-7;
   //inhWeight_[0]=0;
   //inhWeight_[0]=2.5e-7;;
}

ConnStatic::~ConnStatic() {
   if (WSTDP_ != NULL) delete WSTDP_;
   WSTDP_=NULL;
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
   int numNeurons = Simulator::getInstance().getTotalNeurons();
   vector<DistDestNeuron> distDestNeurons[numNeurons];
   int added = 0;
   BGSIZE maxTotalSynapses =  Simulator::getInstance().getMaxSynapsesPerNeuron() * Simulator::getInstance().getTotalNeurons();
   WSTDP_ = new BGFLOAT[maxTotalSynapses];
   

   LOG4CPLUS_INFO(fileLogger_, "Initializing connections");

   for (int srcNeuron = 0; srcNeuron < numNeurons; srcNeuron++) {
      distDestNeurons[srcNeuron].clear();

      // pick the connections shorter than threshConnsRadius
      for (int destNeuron = 0; destNeuron < numNeurons; destNeuron++) {
         if (srcNeuron != destNeuron) {
            BGFLOAT dist = (*layout->dist_)(srcNeuron, destNeuron);
            if (dist <= threshConnsRadius_) {
               DistDestNeuron distDestNeuron;
               distDestNeuron.dist = dist;
               distDestNeuron.destNeuron = destNeuron;
               distDestNeurons[srcNeuron].push_back(distDestNeuron);
            }
         }
      }

      // sort ascendant
      sort(distDestNeurons[srcNeuron].begin(), distDestNeurons[srcNeuron].end());
      // pick the shortest m_nConnsPerNeuron connections
      for (BGSIZE i = 0; i < distDestNeurons[srcNeuron].size() && (int) i < connsPerNeuron_; i++) {
         int destNeuron = distDestNeurons[srcNeuron][i].destNeuron;
         synapseType type = layout->synType(srcNeuron, destNeuron);
         BGFLOAT *sumPoint = &(dynamic_cast<AllNeurons *>(neurons)->summationMap_[destNeuron]);

        // LOG4CPLUS_DEBUG(fileLogger_, "Source: " << srcNeuron << " Dest: " << destNeuron << " Dist: "
         //                                        << distDestNeurons[srcNeuron][i].dist);

         BGSIZE iSyn;
         //ADD ISYN
         synapses->addSynapse(iSyn, type, srcNeuron, destNeuron, sumPoint, Simulator::getInstance().getDeltaT());
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


string weight_str="";
   for(int i=0; i<maxTotalSynapses; i++)
   {
      WSTDP_[i]=dynamic_cast<AllSynapses *>(synapses)->W_[i];
      if(WSTDP_[i]!=0)
        // LOG4CPLUS_DEBUG(synapseLogger_,i << WSTDP_[i]);
         weight_str+=to_string(WSTDP_[i])+" ";
   }
   LOG4CPLUS_DEBUG(synapseLogger_, " "<<weight_str);
   


   int nRewiring = added * rewiringProbability_;

   LOG4CPLUS_DEBUG(fileLogger_,"Rewiring connections: " << nRewiring);

   LOG4CPLUS_DEBUG(fileLogger_,"Added connections: " << added);
}

/*
 * Load member variables from configuration file.
 * Registered to OperationManager as Operations::op::loadParameters
 */
void ConnStatic::loadParameters() {
   ParameterManager::getInstance().getBGFloatByXpath("//threshConnsRadius/text()", threshConnsRadius_);
   ParameterManager::getInstance().getBGFloatByXpath("//connsPerNeuron/text()", connsPerNeuron_);
   ParameterManager::getInstance().getBGFloatByXpath("//rewiringProbability/text()", rewiringProbability_);
   //ParameterManager::getInstance().getBGFloatByXpath("//excWeight/min/text()", excWeight_[0]);
   //ParameterManager::getInstance().getBGFloatByXpath("//excWeight/max/text()", excWeight_[1]);
   //ParameterManager::getInstance().getBGFloatByXpath("//inhWeight/min/text()", inhWeight_[0]);
   //ParameterManager::getInstance().getBGFloatByXpath("//inhWeight/max/text()", inhWeight_[1]);
}


/**
 *  Prints out all parameters to logging file.
 *  Registered to OperationManager as Operation::printParameters
 */
void ConnStatic::printParameters() const {
   LOG4CPLUS_DEBUG(fileLogger_, "CONNECTIONS PARAMETERS" << endl
    << "\tConnections Type: ConnStatic" << endl
    << "\tConnection radius threshold: " << threshConnsRadius_ << endl
    << "\tConnections per neuron: " << connsPerNeuron_ << endl
    << "\tRewiring probability: " << rewiringProbability_ << endl 
    << "\tExhitatory min weight: " << excWeight_[0] << endl 
    << "\tExhitatory max weight: " << excWeight_[1] << endl 
    << "\tInhibitory min weight: " << inhWeight_[0] << endl 
    << "\tInhibitory max weight: " << inhWeight_[1] << endl 
    << endl);
}

