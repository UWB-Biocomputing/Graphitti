//
// Created by chris on 6/26/2020.
//

#pragma once

#include "IFunctionNode.h"
#include "GenericFunctionNode.h"
#include "Operations.h"
#include <list>
#include <iterator>

/**
 * Singleton instance method that registers and executes functions based on operation types.
 *
 * This class allows high level classes to perform high level operations that are defined by lower level classes.
 * Implementation of chain of responsibility design pattern.
 *
 */

class OperationManager {
public:
    // Get Instance method that creates an instance if it doesn't exist, returns the instance of the singleton object
    static OperationManager *getInstance();

    // Takes in a operation type and invokes all registered functions that are registered as that operation type
    bool executeOperation(const Operations::op &operation);

    // Called by lower level classes constructors on creation to register their operations categorized by
    // the operation type
    // Handles function signature: void ()
    bool registerOperation(const Operations::op &operation, function<void()> function);

private:
    // Constructor is private to keep a singleton instance of this class
    OperationManager() {}

    // helper function that adds the registered function as part of the specified operation type
    bool registerOperationHelper(const Operations::op &operation, IFunctionNode *newNode);

    // Singleton instance, reference to this class
    static OperationManager *instance;

    // LinkedLists of functions based on operation type
    list<IFunctionNode *> allocateMemoryList;
    list<IFunctionNode *> deallocateMemoryList;
    list<IFunctionNode *> restoreToDefaultList;
    list<IFunctionNode *> copyToGPUList;
    list<IFunctionNode *> copyFromGPUList;
};


