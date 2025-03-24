/**
 * @file RecordableBase.h
 *
 * @ingroup Simulator/Recorders
 *
 * @brief An abstract base class defining the interface for recordable variables in the simulator.
 *
 * The `RecordableBase` class serves as the base class for all variables that need to be recorded
 * during simulation. It defines the interface for accessing information about recorded variables.
 *     -# the basic data type of recorded variables.
 *     -# the data stored in recorded variables.
 *     -# the number of data in the recorded variable.
 */
#pragma once

using namespace std;
#include "VertexType.h"
#include <string>
#include <typeinfo>
#include <variant>
#include <vector>
// cereal
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>

/// A list of pre-defined basic data types for variablse in all the simulations
/// These pre-defined types should match with the types of variant in Recorder
using variantTypes = variant<uint64_t, bool, int, BGFLOAT, vertexType, double, unsigned char>;

class RecordableBase {
public:
   virtual ~RecordableBase() = default;

   /// Get the number of elements that needs to be recorded
   virtual int getNumElements() const = 0;

   /// Get the value of the recordable variable at the specified index.
   /// @param index The index of the recorded value to retrieve.
   /// @return A variant representing the recorded value.
   virtual variantTypes getElement(int index) const = 0;

   /// Start a new epoch for the recordable variable.
   /// Makes the variable look empty from recorder point of view
   /// Depends on subclass implementation 
   /// Called at the beginning of each simulation epoch to prepare for recording new events.
   virtual void startNewEpoch() = 0;

   /// Set up a string representing the basic data type
   virtual void setDataType() = 0;

   /// Get A string representing the data type of the recordable variable
   /// Dynamic or runtime type information of basic data type
   virtual const string &getDataType() const = 0;

   ///  Cereal serialization method
   template <class Archive> void serialize(Archive &archive)
   {
      archive(cereal::make_nvp("basicDataType", basicDataType_));
   }

protected:
   /// prevents any code outside this class from creating a RecordableBase object
   RecordableBase() = default;
   /// the basic data type in the recorded variable
   std::string basicDataType_;
};
