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

// Get Instance method that creates an instance one doesn't already exist, returns the instance of the singleton object
OperationManager *OperationManager::getInstance() {
    if (instance == nullptr) {
        instance = new OperationManager();
    }
    return instance;
}

// Called by lower level classes constructors on creation to register their operations categorized by the operation type
// This method can be overloaded to handle different function signatures
// Handles function signature: void ()
bool OperationManager::registerOperation(const Operations::op &operation, const function<void()> function) {
    IFunctionNode *chainNode = new GenericFunctionNode(function);
    return registerOperationHelper(operation, chainNode);
}

// Takes in a operation type and invokes all registered functions that are registered as that operation type
bool OperationManager::executeOperation(const Operations::op &operation) {
    list<IFunctionNode *> *listToExecute;
    switch (operation) {
        case Operations::op::allocateMemory :
            listToExecute = &allocateMemoryList;
            break;
        case Operations::op::deallocateMemory :
            listToExecute = &deallocateMemoryList;
            break;
        case Operations::op::restoreToDefault :
            listToExecute = &restoreToDefaultList;
            break;
        case Operations::op::copyToGPU :
            listToExecute = &copyToGPUList;
            break;
        case Operations::op::copyFromGPU :
            listToExecute = &copyFromGPUList;
            break;
        default:
            return false;
    }
    if (listToExecute->size() > 0) {
        list<IFunctionNode *>::iterator i;
        for (i = listToExecute->begin(); i != listToExecute->end(); ++i) {
            (*i)->invokeFunction();
        }
        return true;
    }
    return false;
}

// private function: helper method that adds the registered function as part of the specified operation type
bool OperationManager::registerOperationHelper(const Operations::op &operation, IFunctionNode *newNode) {
    list<IFunctionNode *> *listToAddNode;
    switch (operation) {
        case Operations::op::allocateMemory :
            listToAddNode = &allocateMemoryList;
            break;
        case Operations::op::deallocateMemory :
            listToAddNode = &deallocateMemoryList;
            break;
        case Operations::op::restoreToDefault :
            listToAddNode = &restoreToDefaultList;
            break;
        case Operations::op::copyToGPU :
            listToAddNode = &copyToGPUList;
            break;
        case Operations::op::copyFromGPU :
            listToAddNode = &copyFromGPUList;
            break;
        default:
            return false;
    }

    listToAddNode->push_back(newNode);
    return true;
}
