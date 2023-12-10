/**
 * @file RecordableBase.h
 *
 * @ingroup Simulator/Recorders
 *
 * @brief An abstract base class defining the interface for recordable variables in the simulator.
 *
 * The `RecordableBase` class serves as the base class for all variables that need to be recorded
 * during simulation. It defines the interface for accessing information about recorded variables.
 *     -# the basic data type of recorded variables.
 *     -# the data stored in recorded variables.
 *     -# the number of data in the recorded variable.
 */
#pragma once
#include <variant>
#include <vector>

class RecordableBase {
   // TODO: will change function names later
public:
   virtual ~RecordableBase() = default;
   // Get the number of events in the current epoch for the recordable variable.
   virtual int getNumEventsInEpoch() const = 0;

   // @brief Start a new epoch for the recordable variable.
   // This method is called at the beginning of each simulation epoch to prepare for recording new events.
   virtual void startNewEpoch() = 0;

   // @brief Get the value of the recordable variable at the specified index.
   // @param index The index of the recorded value to retrieve.
   // @return A variant representing the recorded value (uint64_t, double, or string).
   virtual std::variant<uint64_t, double, string> getElement(int index) const = 0;

   // set up a string representing the basic data type
   virtual void setDataType() = 0;

   // @brief Get A string representing the data type of the recordable variable
   virtual string getDataType() const = 0;
};