#pragma once

#include "RecordableBase.h"
#include <stdexcept>   // for std::out_of_range
#include <vector>
// Cereal
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>

template <typename T> class RecordableVector : public RecordableBase, public std::vector<T> {
public:
   RecordableVector()
   {
      setDataType();
   }

   /// Get the number of events in the current epoch for the recordable variable
   int getNumElements() const override
   {
      return std::vector<T>::size();
   }

   /// Start a new epoch for the recordable variable.
   void startNewEpoch() override
   {
      // Implementation for starting a new epoch if needed
   }

   /// Get the value of the recordable variable at the specified index.
   /// @param index The index of the recorded value to retrieve.
   /// @return A variant representing the recorded value.
   variantTypes getElement(int index) const override
   {
      // Return the element at the specified index
      if (index >= 0 && index < std::vector<T>::size()) {
         return vector<T>::operator[](index);
      } else {
         throw std::out_of_range("Index out of range");
      }
   }

   /// Set up a string representing the basic data type
   /// Return a run time type info
   void setDataType() override
   {
      basicDataType_ = typeid(T).name();
   }

   /// Get the data type info in the object at run time
   const std::string &getDataType() const override
   {
      return basicDataType_;
   }

   template <class Archive> void serialize(Archive &archive)
   {
      archive(cereal::base_class<RecordableBase>(this), cereal::base_class<std::vector<T>>(this));
   }
};

// Explicit specialization for RecordableVector<float>
template <> template <class Archive> void RecordableVector<BGFLOAT>::serialize(Archive &archive)
{
   archive(cereal::base_class<RecordableBase>(this),
           cereal::base_class<std::vector<BGFLOAT>>(this));
}

CEREAL_REGISTER_TYPE(RecordableVector<BGFLOAT>);