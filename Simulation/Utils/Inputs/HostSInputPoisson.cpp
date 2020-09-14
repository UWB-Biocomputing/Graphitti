/*
 *      \file HostSInputPoisson.cpp
 *
 *      \author Fumitaka Kawasaki
 *
 *      \brief A class that performs stimulus input (implementation Poisson).
 */

#include "HostSInputPoisson.h"
#include "Core/CPUSpikingModel.h"
#include "tinyxml.h"

/*
 * The constructor for HostSInputPoisson.
 *
 * @param[in] psi       Pointer to the simulation information
 * @param[in] parms     TiXmlElement to examine.
 */
HostSInputPoisson::HostSInputPoisson(TiXmlElement* parms) : SInputPoisson(parms)
{
    
}

/*
 * destructor
 */
HostSInputPoisson::~HostSInputPoisson()
{
}

/*
 * Initialize data.
 *
 * @param[in] psi       Pointer to the simulation information.
 */
void HostSInputPoisson::init()
{
    SInputPoisson::init();

    if (fSInput == false)
        return;
}

/*
 * Terminate process.
 *
 * @param[in] psi       Pointer to the simulation information.
 */
void HostSInputPoisson::term()
{
    SInputPoisson::term();
}

/*
 * Process input stimulus for each time step.
 * Apply inputs on summationPoint.
 *
 * @param[in] psi             Pointer to the simulation information.
 */
void HostSInputPoisson::inputStimulus()
{
    if (fSInput == false)
        return;

#if defined(USE_OMP)
int chunk_size = psi->totalNeurons / omp_get_max_threads();
#endif

#if defined(USE_OMP)
#pragma omp parallel for schedule(static, chunk_size)
#endif
    for (int neuronIndex = 0; neuronIndex < Simulator::getInstance().getTotalNeurons(); neuronIndex++)
    {
        if (masks[neuronIndex] == false)
            continue;

        BGSIZE iSyn = Simulator::getInstance().getMaxSynapsesPerNeuron() * neuronIndex;
        if (--nISIs[neuronIndex] <= 0)
        {
            // add a spike
            dynamic_cast<AllSpikingSynapses*>(synapses_)->preSpikeHit(iSyn);

            // update interval counter (exponectially distribution ISIs, Poisson)
            BGFLOAT isi = -lambda * log(rng.inRange(0, 1));
            // delete isi within refractoriness
            while (rng.inRange(0, 1) <= exp(-(isi*isi)/32))
                isi = -lambda * log(rng.inRange(0, 1));
            // convert isi from msec to steps
            nISIs[neuronIndex] = static_cast<int>( (isi / 1000) / Simulator::getInstance().getDeltaT() + 0.5 );
        }
        // process synapse
        synapses_->advanceSynapse(iSyn, NULL);
    }
}
