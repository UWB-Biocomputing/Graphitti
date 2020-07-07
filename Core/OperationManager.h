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

    // Destructor
    ~OperationManager();

    // Takes in a operation type and invokes all registered functions that are registered as that operation type
    bool executeOperation(const Operations::op &operation);

    // Called by lower level classes constructors on creation to register their operations categorized by
    // the operation type
    // Handles function signature: void ()
    void registerOperation(const Operations::op &operation, function<void()> function);

private:
    // Constructor is private to keep a singleton instance of this class
    OperationManager();

    // Singleton instance, reference to this class
    static OperationManager *instance;

    // LinkedLists of functions based on operation type
    list<IFunctionNode *> *functionList;
};


