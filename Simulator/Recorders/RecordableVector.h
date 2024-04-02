#pragma once
#include "RecordableBase.h"
#include <stdexcept>   // for std::out_of_range
#include <vector>

template <typename T> class RecordableVector : public RecordableBase, public vector<T> {
public:
   RecordableVector ()
   {
      setDataType();
   }
   /// Get the number of events in the current epoch for the recordable variable
   int getNumElements() const override
   {
      return vector<T>::size();
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
      if (index >= 0 && index < vector<T>::size()) {
         return vector<T>::operator[](index);
         // return (*this)[index];
      } else {
         // Handle index out of range
         throw out_of_range("Index out of range");
      }
   }

   /// Set up a string representing the basic data type
   /// Return a run time type info
   void setDataType() override
   {
      basicDataType_ = typeid(T).name();
   }

   /// Get the data type info in the object at run time
   const string &getDataType() const override
   {
      return basicDataType_;
   }
};
