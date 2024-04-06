/**
 * @file Recordable.h
 *
 * @brief A template class providing a concrete implementation of the recordable variable interface.
 *
 * The `Recordable` class template extends the `RecordableBase` interface and provides a concrete
 * implementation for recording variables during simulation. It is templated to accommodate different
 * data types for recording. The class keeps track of recorded values and allows access to information
 * about the recorded variable. Note that it is not intended to directly instantiate a `Recordable` object 
 * for recording; instead, it serves as a base for specific recordable variable implementations.
 */

#pragma once
#include "RecordableBase.h"
#include <string>
#include <vector>
// cereal
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

template <typename T> class Recordable : public RecordableBase {
public:
   /// Constructor
   Recordable()
   {
      setDataType();
   }

   /// Set up a string representing the basic data type
   /// "undefined" means it is not intended to instantiate
   /// a Recordable object for recording
   virtual void setDataType() override
   {
      basicDataType_ = "undefined";
   }

   /// Get A string representing the data type of the recordable variable
   virtual const std::string &getDataType() const override
   {
      return basicDataType_;
   }

   /// Get the number of events in the current epoch for the recordable variable.
   virtual int getNumElements() const override
   {
      return eventTimeSteps_.size();
   }

   /// Start a new epoch for the recordable variable.
   virtual void startNewEpoch() override
   {
      return eventTimeSteps_.clear();
   }

   /// Get the value of the recordable variable at the specified index.
   /// return A variant representing the recorded value (uint64_t, double, or string)
   virtual variantTypes getElement(int index) const override
   {
      return eventTimeSteps_[index];
   }

   ///  Cereal serialization method
   template <class Archive> void serialize(Archive &archive)
   {
      archive(cereal::virtual_base_class<RecordableBase>(this),
              cereal::make_nvp("eventTimeSteps_", eventTimeSteps_));
   }

protected:
   /// Holds the event time steps
   vector<T> eventTimeSteps_;
};

CEREAL_REGISTER_TYPE(Recordable<uint64_t>);
