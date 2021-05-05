#pragma once

#include "Simulation/Core/Operations.h"

/**
 *  @file IFunctionNode.h
 *
 *  @brief Interface for storing and invoking functions. Used to support different FunctionNode classes that
 *  define different function signatures.
 *
 *  @ingroup Core/FunctionNodes
 */

using namespace std;

class IFunctionNode {
public:
    virtual bool invokeFunction(const Operations::op &operation) const = 0;

protected:
    Operations::op operationType_;
};
