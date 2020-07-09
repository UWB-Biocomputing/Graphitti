//
// Created by chris on 6/29/2020.
//

#pragma once

#include "Operations.h"
#include <iostream>

/**
 * Interface for storing and invoking functions.
 * Used to support different FunctionNode classes that define different function signatures.
 */

using namespace std;

class IFunctionNode {
public:
    virtual bool invokeFunction(const Operations::op &operation) = 0;

protected:
    Operations::op operationType;
};
