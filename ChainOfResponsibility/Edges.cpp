//
// Created by chris on 6/23/2020.
//

#include "Edges.h"
#include "IChainNode.h"


Edges::Edges() {}

IChainNode *Edges::SetNextNode(IChainNode *nextNode) {
    this->nextNode = nextNode;
    return nextNode;
}

void Edges::PerformOperation(const Operations::op &operation) {

}