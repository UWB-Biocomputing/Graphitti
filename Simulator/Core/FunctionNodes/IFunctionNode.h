/**
 *  @file IFunctionNode.h
 *
 *  @brief Interface for storing and invoking functions. Used to support different FunctionNode classes that
 *  define different function signatures.
 *
 *  @ingroup Simulator/Core/FunctionNodes
 */

#pragma once

#include "Operations.h"

using namespace std;

class IFunctionNode {
public:
    /// Destructor.
    virtual ~IFunctionNode() {}

    /// Invokes the stored function if the sent operation type matches the operation type the function is stored as.
    virtual bool invokeFunction(const Operations::op &operation) const = 0;

protected:
    /// The operation type of the stored function.
    Operations::op operationType_;
};
