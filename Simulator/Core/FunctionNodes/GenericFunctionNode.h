/**
 *  @file GenericFunctionNode.h
 * 
 *  @ingroup Simulator/Core/FunctionNodes
 *
 *  @brief Stores a function to invoke. Used by operation manager to store functions to defined by an operation type.
 *
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
    std::function<void()> function_; ///< Stored function.
};


