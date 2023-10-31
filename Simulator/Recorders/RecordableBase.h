/**
 * @file RecordableBase.h
 *
 * @ingroup Simulator/Recorders
 *
 * @brief A base interface for Recordable
 *
 * The Recordablebase provides a mechanism for recorder records different types of
 * Recordable template with different types.
 *     -# the neuron ID.
 *     -# time steps of events produced by each neuron,
 */
#pragma once
#include <vector>
#include <variant>

class RecordableBase {

public:
   // TODO: will change function names later

   virtual int getNumEventsInEpoch() const = 0;
   virtual void startNewEpoch() = 0;
   virtual std::variant<uint64_t, double, string> getElement(int index) const = 0;
};