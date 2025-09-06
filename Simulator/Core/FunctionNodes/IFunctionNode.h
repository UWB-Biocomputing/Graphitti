/**
 *  @file IFunctionNode.h
 *
 *  @brief Interface for storing and invoking functions. Used to support different FunctionNode classes that
 *  define different function signatures.
 *
 *  @ingroup Simulator/Core/FunctionNodes
 */

#pragma once

#include "Operations.h"
#include <cstdint> ///for uint64_t

using namespace std;

class IFunctionNode {
public:
   /// Destructor.
   virtual ~IFunctionNode() = default;

   /// TODO: Need to refactor to allow for passing in arguments. Otherwise, FunctionNode classes can not support
   /// non-empty signatures.
   /// Invokes the stored function if the sent operation type matches the operation type the function is stored as.
   virtual bool invokeFunction(const Operations &operation) const = 0;

   /// Invokes the stored function using the two arguments as input
   virtual bool invokeFunction(const Operations &operation, uint64_t arg1, uint64_t arg2) const = 0;

protected:
   /// The operation type of the stored function.
   Operations operationType_;
};
