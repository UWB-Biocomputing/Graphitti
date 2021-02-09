/**
 *  @file GenericFunctionNode.cpp
 *
 *  @brief Stores a function to invoke. Used by operation manager to store functions to defined by an operation type.
 *  Function Signature supported : void ()
 *
 *  @ingroup Simulator/Core/FunctionNodes
 */

#include "GenericFunctionNode.h"

#include <functional>

#include "Simulator/Core/Operations.h"

/// Constructor, Function Signature: void ()
GenericFunctionNode::GenericFunctionNode(const Operations::op &operation, const std::function<void()> &func) {
    operationType_ = operation;
    function_ = func;
}

/// Invokes the stored function if the sent operation type matches the operation type the function is stored as.
bool GenericFunctionNode::invokeFunction(const Operations::op &operation) const {
    if (operation == operationType_) {
        __invoke(function_);
        return true;
    }
    return false;
}