//
// Created by chris on 6/22/2020.
//

#ifndef SUMMEROFBRAIN_CHAINNODE_H
#define SUMMEROFBRAIN_CHAINNODE_H

#include "Operations.h"
#include "IChainNode.h"
#include <functional>
#include <iostream>

using namespace std;

// Interface
template <class OperationObject>
class ChainNode : public IChainNode {
public:
    ChainNode(OperationObject operationObject, std::function<void()> function) {
        this->operationObject = &operationObject;
        this->function = function;
    }

    OperationObject *operationObject;

    std::function<void()> function;

    IChainNode *nextNode;

    // Method for setting the nextNode node in the chain
    IChainNode *setNextNode(IChainNode *nextNode) override {
        this->nextNode = nextNode;
    }

    // Generic operation
    void performOperation() {
        __invoke(function);
    }

};


#endif //SUMMEROFBRAIN_CHAINNODE_H
