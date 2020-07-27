/**
 * @file OperationManager.h
 *
 * @brief Singleton instance method that registers and executes functions based on operation types.
 *
 * @ingroup Core
 *
 * This class allows high level classes to perform high level operations that are defined by lower level classes.
 * Implementation of chain of responsibility design pattern.
 *
 * The implementation allows for multi-threaded use.
 */

#pragma once

#include <functional>
#include <list>
#include <memory>

#include "Simulation/Core/FunctionNodes/IFunctionNode.h"
#include "Operations.h"

using namespace std;

class OperationManager {
public:
    /// Get Instance method that returns a reference to this object.
    static OperationManager &getInstance();

    /// Called by lower level classes constructors on creation to register their operations with their operation type
    /// This method can be overloaded to handle different function signatures.
    /// Handles function signature: void ()
    void registerOperation(const Operations::op &operation, function<void()> function);

    /// Takes in a operation type and invokes all registered functions that are classified as that operation type.
    void executeOperation(const Operations::op &operation) const;

    /// Delete these methods because they can cause copy instances of the singleton when using threads.
    OperationManager(OperationManager const &) = delete;
    void operator=(OperationManager const &) = delete;

private:
    /// Constructor is private to keep a singleton instance of this class.
    OperationManager() {}

    /// LinkedLists of functions based on operation type.
    list<unique_ptr<IFunctionNode>> functionList_;
};


