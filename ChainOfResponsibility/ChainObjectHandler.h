//
// Created by chris on 6/26/2020.
//

#ifndef SUMMEROFBRAIN_CHAINOBJECTHANDLER_H
#define SUMMEROFBRAIN_CHAINOBJECTHANDLER_H

#include "IChainNode.h"
#include "Operations.h"

class ChainObjectHandler {
public:
    // Get Instance method that acts as a constructor, returns the instance of the singleton object
    static ChainObjectHandler *getInstance();

    // Method for executing operations in the chain of objects
    void ExecuteOperation(const Operations::op &operation);

    // Look into where to store this information probably a new class


private:
    // Constructor is private since the getInstance serves as an alternate constructor
    ChainObjectHandler() {}

    // Singleton instance, reference to this class
    static ChainObjectHandler *instance;

    // Head of the chain of lower level objects
    IChainNode *head;
};



#endif //SUMMEROFBRAIN_CHAINOBJECTHANDLER_H
