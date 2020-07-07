//
// Created by chris on 6/26/2020.
//

#include "OperationManager.h"
#include "GenericFunctionNode.h"

/**
 * Singleton instance method that registers and executes functions based on operation types.
 *
 * This class allows high level classes to perform high level operations that are defined by lower level classes.
 * Implementation of chain of responsibility design pattern.
 *
 */

// Singleton instance, reference to this class, initialized as nullptr so that it can be accessed
OperationManager *OperationManager::instance = nullptr;

// Private Constructor to keep a singleton instance of this class
OperationManager::OperationManager() {
    functionList = new list<IFunctionNode *>();
}

// Destructor
OperationManager::~OperationManager() {
    delete functionList;
}

// Get Instance method that creates an instance if one doesn't already exist, returns the instance of the singleton object
OperationManager *OperationManager::getInstance() {
    if (instance == nullptr) {
        instance = new OperationManager();
    }
    return instance;
}

// Called by lower level classes constructors on creation to register their operations with their operation type
// This method can be overloaded to handle different function signatures
// Handles function signature: void ()
void OperationManager::registerOperation(const Operations::op &operation, function<void()> function) {
    IFunctionNode *functionNode = new GenericFunctionNode(operation, function);
    functionList->push_back(functionNode);
}

// Takes in a operation type and invokes all registered functions that are classified as that operation type
bool OperationManager::executeOperation(const Operations::op &operation) {
    if (functionList->size() > 0) {
        bool operationExecuted = false;
        list<IFunctionNode *>::iterator i;
        for (i = functionList->begin(); i != functionList->end(); ++i) {
            if ((*i)->invokeFunction(operation)) {
                operationExecuted = true;
            }
        }
        return operationExecuted;
    }
    return false;
}

