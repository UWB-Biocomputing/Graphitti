#pragma once
#include <vector>
#include <variant>

class RecordableBase {

public:
virtual int getNumEventsInEpoch() const = 0;
virtual void startNewEpoch() = 0;
virtual std::variant<uint64_t, double, char> getElement(int index) const = 0;
// virtual string getDataType() const = 0
// virtual ~RecordableBase() {}
};