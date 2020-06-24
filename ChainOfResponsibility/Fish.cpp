//
// Created by chris on 6/23/2020.
//

#include "Fish.h"
#include "IChainNode.h"


Fish::Fish() {}

IChainNode *Fish::SetNextNode(IChainNode *nextNode) {
    this->nextNode = nextNode;
    return nextNode;
}

string Fish::PerformOperation(string request) {
    if (request == "fish") {
        return "There's a fish in the chain";
    } else {
        if (nextNode)
            return nextNode->PerformOperation(request);
        else
            return "Request can't be processed";
    }
}