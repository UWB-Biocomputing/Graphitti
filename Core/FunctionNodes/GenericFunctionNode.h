#pragma once

#include <functional>

#include "IFunctionNode.h"

/**
 * Stores a function to invoke.
 * Used by operation manager to store functions to defined by an operation type.
 *
 * Function Signature supported : void ()
 */

using namespace std;

class GenericFunctionNode : public IFunctionNode {
public:
    // Constructor
    GenericFunctionNode(const Operations::op &operationType, std::function<void()> function);

    // Invokes the stored function if the sent operation type matches the operation type the function is stored as.
    bool invokeFunction(const Operations::op &operation) override;

private:
    // Stored function.
    std::function<void()> function;
};


