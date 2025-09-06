/**
 *  @file TwoUint64ArgFunctionNode.h
 * 
 *  @ingroup Simulator/Core/FunctionNodes
 *
 *  @brief Stores a function with two uint64_t args to invoke. Used by operation manager to store functions to defined by an operation type.
 *
 */

#pragma once

#include "IFunctionNode.h"
#include <functional>

using namespace std;

class TwoUint64ArgFunctionNode : public IFunctionNode {
public:
   /// Constructor, Function Signature: void ()
   TwoUint64ArgFunctionNode(const Operations &operationType, const std::function<void(uint64_t,uint64_t)> &function);

   /// Destructor
   ~TwoUint64ArgFunctionNode() = default;

   /// TODO: Remove when IFunctionNode supports functions with non-empty signatures
   virtual bool invokeFunction(const Operations &operation) const { return false; }

   /// Invokes the stored function if the sent operation type matches the operation type the function is stored as.
   virtual bool invokeFunction(const Operations &operation, uint64_t arg1, uint64_t arg2) const override;

private:
   std::function<void(uint64_t,uint64_t)> function_;   ///< Stored function.
};
