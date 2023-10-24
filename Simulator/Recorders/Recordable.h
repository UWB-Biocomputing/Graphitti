#pragma once
#include <vector>
#include "RecordableBase.h"

template<typename T>
class Recordable : public RecordableBase{
protected:
   std::vector<T> eventTimeSteps_;

public:
// vector<T> eventTimeSteps_;
// 	virtual int getNumEventsInEpoch() const = 0;
   virtual int getNumEventsInEpoch() const{
      return eventTimeSteps_.size();
   }
   virtual void startNewEpoch(){
      return eventTimeSteps_.clear();
   }
   virtual std::variant<uint64_t, double, char> getElement(int index) const{
      return eventTimeSteps_[index];
   }
   // virtual ~RecordableBase() {}
   // Access an element of the vector using the [] operator
   T operator[](int index) const{
      return eventTimeSteps_[index];
   }
};
