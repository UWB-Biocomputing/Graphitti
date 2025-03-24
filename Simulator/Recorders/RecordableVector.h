/**
 * @file RecordableVector.h
 *
 * @brief A template class providing a concrete implementation of the recordable variable interface for one-dimensional variables.
 *
 * The `RecordableVector` class template extends the `RecordableBase` interface and provides a concrete
 * implementation for recording one-dimensional variables during simulation. It is templated to accommodate different
 * data types for recording. The class maintains a vector of recorded values and allows access to information
 * about the recorded variable. This class should be instantiated when there is a one-dimensional variable 
 * that needs to be recorded by the recorder.
 */

#pragma once
#include "RecordableBase.h"
#include <string>
#include <vector>
// cereal
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

template <typename T> class RecordableVector : public RecordableBase {
public:
   /// Constructor
   RecordableVector()
   {
      setDataType();
   }

   /// Set up a string representing the basic data type
   /// Return a run time type info
   virtual void setDataType() override
   {
      basicDataType_ = typeid(T).name();
   }

   /// Get a string representing the data type of the recordable variable
   virtual const std::string &getDataType() const override
   {
      return basicDataType_;
   }

   /// Get the number of elements that needs to be recorded
   virtual int getNumElements() const override
   {
      return dataSeries_.size();
   }

   /// Start a new epoch for the recordable variable.
   virtual void startNewEpoch() override
   {
      return dataSeries_.clear();
   }

   /// Get the value of the recordable variable at the specified index.
   /// return A variant representing the recorded value (uint64_t, double, or string)
   virtual variantTypes getElement(int index) const override
   {
      return dataSeries_[index];
   }

   /// Resize the eventTimeSteps vector to the specified size.
   virtual void resize(int maxEvents)
   {
      dataSeries_.resize(maxEvents);
   }

   /// Assigns the given value to the vector for the specified size. 
   virtual void assign(size_t size, const T& value)
   {
      dataSeries_.assign(size, value);
   }

   /// Overload the operator to set the value at a specific index
   T &operator[](int index)
   {
      if (index >= 0 && index < dataSeries_.size()) {
         return dataSeries_[index];
      } else {
         throw std::out_of_range("Index out of range");
      }
   }

   /// Add a new value to recordable vector 
   void push_back(const T &value)
   {
      dataSeries_.push_back(value);
   }

   /// Method to retrieve the underlying std::vector<T>
   const std::vector<T> &getVector() const
   {
      return dataSeries_;
   }

   ///  Cereal serialization method
   template <class Archive> void serialize(Archive &archive)
   {
      archive(cereal::virtual_base_class<RecordableBase>(this),
              cereal::make_nvp("dataSeries", dataSeries_));
   }

protected:
   /// Holds the event time steps
   vector<T> dataSeries_;
};

CEREAL_REGISTER_TYPE(RecordableVector<BGFLOAT>);
