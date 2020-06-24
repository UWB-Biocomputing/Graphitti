//
// Created by chris on 6/22/2020.
//

#include "Dog.h"
#include "IChainNode.h"

using namespace std;

Dog::Dog() {}

IChainNode *Dog::SetNextNode(IChainNode *nextNode) {
    this->nextNode = nextNode;
    return nextNode;
}

string Dog::PerformOperation(string request) {
    if (request == "dog") {
        return "There's a dog in the chain";
    } else {
        if (nextNode)
            return nextNode->PerformOperation(request);
        else
            return "Request can't be processed";
    }
}