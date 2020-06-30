//
// Created by chris on 6/26/2020.
//

#pragma once

#include "IChainNode.h"
#include "ChainNode.h"
#include "Operations.h"
#include <list>
#include <iterator>

class ChainOperationManager {
public:
    // Get Instance method that acts as a constructor, returns the instance of the singleton object
    static ChainOperationManager *getInstance();

    // Method for executing operations in the chain of objects
    bool executeOperation(const Operations::op &operation);

    bool addNodeToChain(const Operations::op &operation, IChainNode *newNode);

private:
    // Constructor is private since the getInstance serves as an alternate constructor
    ChainOperationManager() {}

    // Singleton instance, reference to this class
    static ChainOperationManager *instance;

    list<IChainNode*> allocateMemoryList;
    list<IChainNode*> deallocateMemoryList;
    list<IChainNode*> restoreToDefaultList;
    list<IChainNode*> copyToGPUList;
    list<IChainNode*> copyFromGPUList;
};


