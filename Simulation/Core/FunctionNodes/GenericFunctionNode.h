/**
 *  @file GenericFunctionNode.h
 *
 *  @brief Stores a function to invoke. Used by operation manager to store functions to defined by an operation type.
 *  Function Signature supported : void ()
 *
 *  @ingroup FunctionNodes
 */

#pragma once

#include <functional>

#include "IFunctionNode.h"

using namespace std;

class GenericFunctionNode : public IFunctionNode {
public:
    /// Constructor, Function Signature: void ()
    GenericFunctionNode(const Operations::op &operationType, const std::function<void()> &function);

    /// Destructor
    ~GenericFunctionNode();

    /// Invokes the stored function if the sent operation type matches the operation type the function is stored as.
    bool invokeFunction(const Operations::op &operation) const override;

private:
    /// Stored function.
    std::function<void()> function_;
};


