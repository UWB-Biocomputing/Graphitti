/**
 *  @file TwoUint64ArgFunctionNode.cpp
 * 
 *  @ingroup Simulator/Core/FunctionNodes
 *
 *  @brief Stores a function with two uint64_t args to invoke. Used by operation manager to store functions to defined by an operation type.
 * 
 *  Function Signature supported : void (uint64_t,uint64_t)
 *
 */

#include "TwoUint64ArgFunctionNode.h"
#include "Operations.h"
#include <functional>

/// Constructor, Function Signature: void (uint64_t, uint64_t)
TwoUint64ArgFunctionNode::TwoUint64ArgFunctionNode(const Operations &operation,
                                         const std::function<void(uint64_t,uint64_t)> &func)
{
   operationType_ = operation;
   function_ = func;
}

/// Invokes the stored function if the sent operation type matches the operation type the function is stored as.
bool TwoUint64ArgFunctionNode::invokeFunction(const Operations &operation, uint64_t arg1, uint64_t arg2) const
{
   if (operation == operationType_) {
      __invoke(function_, arg1, arg2);
      return true;
   }
   return false;
}