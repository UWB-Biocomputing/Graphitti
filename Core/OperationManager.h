#pragma once

#include <functional>
#include <list>
#include <memory>

#include "IFunctionNode.h"
#include "Operations.h"

/**
 * Singleton instance method that registers and executes functions based on operation types.
 *
 * This class allows high level classes to perform high level operations that are defined by lower level classes.
 * Implementation of chain of responsibility design pattern.
 *
 * The implementation allows for multithreaded use.
 */

using namespace std;

class OperationManager {
public:
    // Get Instance method that returns a reference to this object.
    static OperationManager &getInstance();

    // Called by lower level classes constructors on creation to register their operations categorized by
    // the operation type.
    // Handles function signature: void ()
    void registerOperation(const Operations::op &operation, function<void()> function);

    // Takes in a operation type and invokes all registered functions that are registered as that operation type.
    void executeOperation(const Operations::op &operation);

    // Delete these methods because they can cause multiple instances of the singleton when using threads.
    OperationManager(OperationManager const &) = delete;
    void operator=(OperationManager &)  = delete;

private:
    // Constructor is private to keep a singleton instance of this class.
    OperationManager() {}

    // LinkedLists of functions based on operation type.
    list<unique_ptr<IFunctionNode>> functionList;
};


