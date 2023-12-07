/**
 * @file Recordable.h
 *
 * @brief A template class providing a concrete implementation of the recordable variable interface.
 *
 * The `Recordable` class template extends the `RecordableBase` interface and provides a concrete
 * implementation for recording variables during simulation. It is templated to accommodate different
 * data types for recording. The class keeps track of recorded values and allows access to information
 * about the recorded variable.
 */

#pragma once
#include "RecordableBase.h"
#include <vector>

template <typename T> class Recordable : public RecordableBase {
protected:
   /// Holds the event time steps
   std::vector<T> eventTimeSteps_;
   // Private members to store the basic data type for recorded values
   std::string basicDataType_;

public:
   // Constructor
   Recordable() : basicDataType_("variant")
   {
   }

   // Get the number of events in the current epoch for the recordable variable.
   virtual int getNumEventsInEpoch() const
   {
      return eventTimeSteps_.size();
   }

   // @brief Start a new epoch for the recordable variable.
   virtual void startNewEpoch()
   {
      return eventTimeSteps_.clear();
   }

   // @brief Get the value of the recordable variable at the specified index.
   // return A variant representing the recorded value (uint64_t, double, or string)
   virtual std::variant<uint64_t, double, string> getElement(int index) const
   {
      return eventTimeSteps_[index];
   }

   // @brief Get A string representing the data type of the recordable variable
   virtual string getDataType() const
   {
      return basicDataType_;
      ;
   }
};
