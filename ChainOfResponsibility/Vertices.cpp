//
// Created by chris on 6/22/2020.
//

#include "Vertices.h"
#include "IChainNode.h"

using namespace std;

Vertices::Vertices() {}

IChainNode *Vertices::SetNextNode(IChainNode *nextNode) {
    this->nextNode = nextNode;
    return nextNode;
}

void Vertices::PerformOperation(const Operations::op &operation) {
    switch(operation) {
        case Operations::op::allocateMemory : 
            allocateMemory();
            break;


    }
    if (nextNode != nullptr) {
        nextNode->PerformOperation(operation);
    }
}

void Vertices::allocateMemory() {

}