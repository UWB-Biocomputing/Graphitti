//
// Created by chris on 6/26/2020.
//

#include "ChainOperationManager.h"
#include "ChainNode.h"

// Singleton instance, reference to this class, initialized as nullptr so that it can be accessed
ChainOperationManager *ChainOperationManager::instance = nullptr;

// Get Instance method that acts as a constructor, returns the instance of the singleton object
ChainOperationManager *ChainOperationManager::getInstance() {
    if (instance == nullptr) {
        instance = new ChainOperationManager();
    }
    return instance;
}

// Method for executing operations in the chain of objects
bool ChainOperationManager::executeOperation(const Operations::op &operation) {
    list<IChainNode*> *listToExecute;
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
        list<IChainNode*>::iterator i;
        for (i = listToExecute->begin(); i != listToExecute->end(); ++i) {
            (*i)->performOperation();
        }
        return true;
    }
    return false;
}

bool ChainOperationManager::addNodeToChain(const Operations::op &operation, IChainNode *newNode) {
    list<IChainNode*> *listToAddNode;
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
