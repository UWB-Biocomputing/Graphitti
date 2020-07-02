//
// Created by chris on 6/22/2020.
//

#pragma once

#include "Operations.h"
#include "IFunctionNode.h"
#include <functional>
#include <iostream>

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
    GenericFunctionNode(std::function<void()> function);

    // Invokes the stored function
    void invokeFunction() override;

private:
    // Stored function
    std::function<void()> function;
};


