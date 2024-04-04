#pragma once

#include "RecordableBase.h"
#include <stdexcept>   // for std::out_of_range
#include <vector>
//cereal
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>

template <typename T> class RecordableVector : public RecordableBase {
public:
   RecordableVector()
   {
      setDataType();
   }

   void resizeRecordableVector(int size)
   {
      data_.resize(size);
   }

   /// Get the number of events in the current epoch for the recordable variable
   int getNumElements() const override
   {
      return data_.size();
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
      if (index >= 0 && index < getNumElements()) {
         return data_[index];
      } else {
         // Handle index out of range
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
      archive(cereal::virtual_base_class<RecordableBase>(this), data_);
   }

   T getDataAtIndex(int index) const
   {
      return data_[index];
   }

   void setData(int index, const T &value)
   {
      if (index >= 0 && index < getNumElements()) {
         data_[index] = value;
      } else {
         throw std::out_of_range("Index out of range");
      }
   }

   const std::vector<T> &getData() const
   {
      return data_;
   }

private:
   vector<T> data_;
};

CEREAL_REGISTER_TYPE(RecordableVector<BGFLOAT>);
