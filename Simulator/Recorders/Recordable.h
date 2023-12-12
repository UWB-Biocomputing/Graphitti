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
public:
   /// Constructor
   Recordable()
   {
      setDataType();
   }

   /// set up a string representing the basic data type
   virtual void setDataType()
   {
      basicDataType_ = "variant";
   }

   /// Get the number of events in the current epoch for the recordable variable.
   virtual int getNumEventsInEpoch() const
   {
      return eventTimeSteps_.size();
   }

   /// Start a new epoch for the recordable variable.
   virtual void startNewEpoch()
   {
      return eventTimeSteps_.clear();
   }

   /// Get the value of the recordable variable at the specified index.
   /// return A variant representing the recorded value (uint64_t, double, or string)
   virtual variant<uint64_t, double, string> getElement(int index) const
   {
      return eventTimeSteps_[index];
   }

   /// Get A string representing the data type of the recordable variable
   virtual string getDataType() const
   {
      return basicDataType_;
   }

protected:
   /// Holds the event time steps
   vector<T> eventTimeSteps_;
   /// Private members to store the basic data type for recorded values
   string basicDataType_;
};
