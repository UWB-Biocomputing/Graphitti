//
// Created by chris on 6/26/2020.
//

#include "OperationManager.h"
#include "GenericFunctionNode.h"
#include <iostream>

/**
 * Singleton instance method that registers and executes functions based on operation types.
 *
 * This class allows high level classes to perform high level operations that are defined by lower level classes.
 * Implementation of chain of responsibility design pattern.
 *
 */

using namespace std;

// Singleton instance, reference to this class, initialized as nullptr so that it can be accessed
OperationManager *OperationManager::instance = nullptr;

// Private Constructor to keep a singleton instance of this class
OperationManager::OperationManager() {}

// Destructor
OperationManager::~OperationManager() {}

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
    try {
        functionList.push_back(unique_ptr<IFunctionNode>(new GenericFunctionNode(operation, function)));
    }
    catch (exception e) {
        throw runtime_error(string(e.what()) + " in OperationManager::registerOperation");
    }
}

// Takes in a operation type and invokes all registered functions that are classified as that operation type
void OperationManager::executeOperation(const Operations::op &operation) {
    if (functionList.size() > 0) {
        for (auto i = functionList.begin(); i != functionList.end(); ++i) {
            (*i)->invokeFunction(operation);
        }
    }
}

