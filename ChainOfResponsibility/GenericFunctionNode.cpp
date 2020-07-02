//
// Created by chris on 6/30/2020.
//

#include "GenericFunctionNode.h"

/**
 * Stores a function to invoke.
 * Used by operation manager to store functions to defined by an operation type.
 *
 * Function Signature supported : void ()
 */

// Constructor
GenericFunctionNode::GenericFunctionNode(std::function<void()> function) {
    this->function = function;
}

// Invokes the stored function
void GenericFunctionNode::invokeFunction() {
    __invoke(function);
}