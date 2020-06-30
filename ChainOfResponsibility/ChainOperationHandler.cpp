//
// Created by chris on 6/26/2020.
//

#include "ChainOperationHandler.h"
#include "ChainNode.h"

// Singleton instance, reference to this class, initialized as nullptr so that it can be accessed
ChainOperationHandler *ChainOperationHandler::instance = nullptr;

// Get Instance method that acts as a constructor, returns the instance of the singleton object
ChainOperationHandler *ChainOperationHandler::getInstance() {
    if (instance == nullptr) {
        instance = new ChainOperationHandler();
    }
    return instance;
}

// Method for executing operations in the chain of objects
void ChainOperationHandler::executeOperation(const Operations::op &operation) {
    head->performOperation();
}

//void ChainOperationHandler::addNodeToChain(ChainNode *newNode) {
//    if (!head) {
//        head = newNode;
//    } else {
//        ChainNode<> *current = head;
//        while (current->nextNode != nullptr) {
//            current = current->nextNode;
//        }
//        current->nextNode = newNode;
//    }
//}
