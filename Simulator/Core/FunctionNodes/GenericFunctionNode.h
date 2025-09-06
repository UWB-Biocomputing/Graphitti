/**
 *  @file GenericFunctionNode.h
 * 
 *  @ingroup Simulator/Core/FunctionNodes
 *
 *  @brief Stores a function to invoke. Used by operation manager to store functions to defined by an operation type.
 *
 */

#pragma once

#include "IFunctionNode.h"
#include <functional>

using namespace std;

class GenericFunctionNode : public IFunctionNode {
public:
   /// Constructor, Function Signature: void ()
   GenericFunctionNode(const Operations &operationType, const std::function<void()> &function);

   /// Destructor
   ~GenericFunctionNode() = default;

   /// Invokes the stored function if the sent operation type matches the operation type the function is stored as.
   virtual bool invokeFunction(const Operations &operation) const override;

   /// TODO: Remove when IFunctionNode supports functions with non-empty signatures
   virtual bool invokeFunction(const Operations &operation, uint64_t arg1, uint64_t arg2) const { return false; }

private:
   std::function<void()> function_;   ///< Stored function.
};
