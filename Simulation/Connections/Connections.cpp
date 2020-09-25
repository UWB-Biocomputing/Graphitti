/* ------------- CONNECTIONS STRUCT ------------ *\
 * Below all of the resources for the various
 * connections are instantiated and initialized.
 * All of the allocation for memory is done in the
 * constructor’s parameters and not in the body of
 * the function. Once all memory has been allocated
 * the constructor fills in known information
 * into “radii” and “rates”.
\* --------------------------------------------- */
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

#include "Connections.h"
#include "IAllSynapses.h"
#include "IAllNeurons.h"
#include "AllSynapses.h"
#include "AllNeurons.h"
#include "OperationManager.h"
#include "ParameterManager.h"
#include "EdgesFactory.h"

Connections::Connections() {
   // Create Edges/Synapses class using type definition in configuration file
   string type;
   ParameterManager::getInstance().getStringByXpath("//SynapsesParams/@class", type);
   synapses_ = EdgesFactory::getInstance()->createEdges(type);

   // Register printParameters function as a printParameters operation in the OperationManager
   function<void()> printParametersFunc = bind(&Connections::printParameters, this);
   OperationManager::getInstance().registerOperation(Operations::printParameters, printParametersFunc);

   // Register loadParameters function with Operation Manager
   function<void()> function = std::bind(&Connections::loadParameters, this);
   OperationManager::getInstance().registerOperation(Operations::op::loadParameters, function);

   // Get a copy of the file logger to use log4cplus macros
   fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));
}

Connections::~Connections() {
}

shared_ptr<IAllSynapses> Connections::getSynapses() const {
   return synapses_;
}

shared_ptr<SynapseIndexMap> Connections::getSynapseIndexMap() const {
   return synapseIndexMap_;
}

void Connections::createSynapseIndexMap() {
   synapseIndexMap_ = shared_ptr<SynapseIndexMap>(synapses_->createSynapseIndexMap());
}

/*
 *  Update the connections status in every epoch.
 *
 *  @param  neurons  The Neuron list to search from.
 *  @param  layout   Layout information of the neunal network.
 *  @return true if successful, false otherwise.
 */
bool Connections::updateConnections(IAllNeurons &neurons, Layout *layout) {
   return false;
}

#if defined(USE_GPU)
void Connections::updateSynapsesWeights(const int numNeurons, IAllNeurons &neurons, IAllSynapses &synapses, AllSpikingNeuronsDeviceProperties* m_allNeuronsDevice, AllSpikingSynapsesDeviceProperties* m_allSynapsesDevice, Layout *layout)
{
}
#else

/*
 *  Update the weight of the Synapses in the simulation.
 *  Note: Platform Dependent.
 *
 *  @param  numNeurons Number of neurons to update.
 *  @param  neurons     The Neuron list to search from.
 *  @param  synapses    The Synapse list to search from.
 */
void Connections::updateSynapsesWeights(const int numNeurons, IAllNeurons &neurons, IAllSynapses &synapses, Layout *layout) {
}

#endif // !USE_GPU

/*
 *  Creates synapses from synapse weights saved in the serialization file.
 *
 *  @param  numNeurons Number of neurons to update.
 *  @param  layout      Layout information of the neunal network.
 *  @param  ineurons    The Neuron list to search from.
 *  @param  isynapses   The Synapse list to search from.
 */
void Connections::createSynapsesFromWeights(const int numNeurons, Layout *layout, IAllNeurons &ineurons,
                                            IAllSynapses &isynapses) {
   AllNeurons &neurons = dynamic_cast<AllNeurons &>(ineurons);
   AllSynapses &synapses = dynamic_cast<AllSynapses &>(isynapses);

   // for each neuron
   for (int iNeuron = 0; iNeuron < numNeurons; iNeuron++) {
      // for each synapse in the neuron
      for (BGSIZE synapseIndex = 0;
           synapseIndex < Simulator::getInstance().getMaxSynapsesPerNeuron(); synapseIndex++) {
         BGSIZE iSyn = Simulator::getInstance().getMaxSynapsesPerNeuron() * iNeuron + synapseIndex;
         // if the synapse weight is not zero (which means there is a connection), create the synapse
         if (synapses.W_[iSyn] != 0.0) {
            BGFLOAT theW = synapses.W_[iSyn];
            BGFLOAT *sumPoint = &(neurons.summationMap_[iNeuron]);
            int srcNeuron = synapses.sourceNeuronIndex_[iSyn];
            int destNeuron = synapses.destNeuronIndex_[iSyn];
            synapseType type = layout->synType(srcNeuron, destNeuron);
            synapses.synapseCounts_[iNeuron]++;
            synapses.createSynapse(iSyn, srcNeuron, destNeuron, sumPoint, Simulator::getInstance().getDeltaT(),
                                   type);
            synapses.W_[iSyn] = theW;
         }
      }
   }
}



