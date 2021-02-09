/**
 *  @file IFunctionNode.h
 *
 *  @brief Interface for storing and invoking functions. Used to support different FunctionNode classes that
 *  define different function signatures.
 *
 *  @ingroup Simulation/Core/FunctionNodes
 */

#pragma once

#include "Simulator/Core/Operations.h"

using namespace std;

class IFunctionNode {
public:
    virtual bool invokeFunction(const Operations::op &operation) const = 0;

protected:
    Operations::op operationType_;
};
