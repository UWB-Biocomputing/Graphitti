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
   virtual int getNumEventsInEpoch() const = 0;
   virtual void startNewEpoch() = 0;
   virtual std::variant<uint64_t, double, char> getElement(int index) const = 0;
   // virtual ~RecordableBase() {}
//   // Access an element of the vector using the [] operator
//    T operator[](int index) const = 0;
};
