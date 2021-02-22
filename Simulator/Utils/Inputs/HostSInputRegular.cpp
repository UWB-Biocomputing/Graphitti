/**
 * @file HostSInputRegular.cpp
 *
 * @ingroup Simulator/Utils/Inputs
 * 
 * @brief A class that performs stimulus input (implementation Regular).
 */

#include "HostSInputRegular.h"

/// constructor
///
/// @param[in] psi       Pointer to the simulation information
/// @param[in] parms     TiXmlElement to examine.
HostSInputRegular::HostSInputRegular(TiXmlElement* parms) : SInputRegular(parms)
{
    
}

HostSInputRegular::~HostSInputRegular()
{
}

/// Initialize data.
///
/// @param[in] psi       Pointer to the simulation information.
void HostSInputRegular::init()
{
    SInputRegular::init();
}

/// Terminate process.
///
/// @param[in] psi       Pointer to the simulation information.
void HostSInputRegular::term()
{
    if (values != NULL)
        delete[] values;

    if (nShiftValues != NULL)
        delete[] nShiftValues;
}

/// Process input stimulus for each time step.
/// Apply inputs on summationPoint.
///
/// @param[in] psi             Pointer to the simulation information.
void HostSInputRegular::inputStimulus()
{
    if (fSInput == false)
        return;

#if defined(USE_OMP)
int chunk_size = psi->totalVertices / omp_get_max_threads();
#endif

#if defined(USE_OMP)
#pragma omp parallel for schedule(static, chunk_size)
#endif
    // add input to each summation point
    for (int i = Simulator::getInstance().getTotalVertices() - 1; i >= 0; --i)
    {
        if ( (nStepsInCycle >= nShiftValues[i]) && (nStepsInCycle < (nShiftValues[i] + nStepsDuration ) % nStepsCycle) )
            Simulator::getInstance().getPSummationMap()[i] += values[i];
    }

    // update cycle count 
    nStepsInCycle = (nStepsInCycle + 1) % nStepsCycle;
}
