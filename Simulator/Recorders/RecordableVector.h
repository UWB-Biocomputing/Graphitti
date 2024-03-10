#pragma once
#include "RecordableBase.h"
#include <stdexcept>   // for std::out_of_range
#include <vector>

template <typename T> class RecordableVector : public RecordableBase, public std::vector<T> {
public:
   /// Get the number of events in the current epoch for the recordable variable
   int getNumElements() const override
   {
      return std::vector<T>::size();
   }

   /// Start a new epoch for the recordable variable.
   void startNewEpoch() override
   {
      // Implementation for starting a new epoch if need
   }

   /// Get the value of the recordable variable at the specified index.
   /// @param index The index of the recorded value to retrieve.
   /// @return A variant representing the recorded value.
   variantTypes getElement(int index) const override
   {
      // Return the element at the specified index
      if (index >= 0 && index < std::vector<T>::size()) {
         return std::vector<T>::operator[](index);
         // return (*this)[index];
      } else {
         // Handle index out of range
         throw std::out_of_range("Index out of range");
      }
   }

   /// Set up a string representing the basic data type
   void setDataType() override
   {
      basicDataType_ = "T";
   }

   /// Get A string representing the data type of the recordable variable
   string &getDataType() override
   {
      // Implementation for getting data type
      return basicDataType_;
   }
};
