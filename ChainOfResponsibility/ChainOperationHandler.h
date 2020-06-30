//
// Created by chris on 6/26/2020.
//

#ifndef SUMMEROFBRAIN_CHAINOPERATIONHANDLER_H
#define SUMMEROFBRAIN_CHAINOPERATIONHANDLER_H

#include "ChainNode.h"
#include "Operations.h"

class ChainOperationHandler {
public:
    // Get Instance method that acts as a constructor, returns the instance of the singleton object
    static ChainOperationHandler *getInstance();

    // Method for executing operations in the chain of objects
    void executeOperation(const Operations::op &operation);

//    void addNodeToChain(ChainNode *newNode);

private:
    // Constructor is private since the getInstance serves as an alternate constructor
    ChainOperationHandler() {}

    // Singleton instance, reference to this class
    static ChainOperationHandler *instance;

    // Head of the chain of lower level objects
    IChainNode *head;
};



#endif //SUMMEROFBRAIN_CHAINOPERATIONHANDLER_H
