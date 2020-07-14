#include "Core/FunctionNodes/GenericFunctionNode.h"

#include <functional>

#include "Core/FunctionNodes/IFunctionNode.h"
#include "Core/Operations.h"

/**
 * Stores a function and the operation type of that function.
 * Used by operation manager to execute operations by invoking functions of a certain operation type
 *
 * Function Signature supported : void ()
 */

// Constructor
GenericFunctionNode::GenericFunctionNode(const Operations::op &operation, std::function<void()> func) {
    operationType = operation;
    function = func;
}

// Invokes the stored function if the sent operation type matches the operation type the function is stored as
bool GenericFunctionNode::invokeFunction(const Operations::op &operation) {
    if (operation == operationType) {
        __invoke(function);
        return true;
    }
    return false;
}