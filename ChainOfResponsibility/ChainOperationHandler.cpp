#include "ChainObjectHandler.h"
#include "IChainNode.h"

// Singleton instance, reference to this class, initialized as nullptr so that it can be accessed
ChainObjectHandler *ChainObjectHandler::instance = nullptr;

// Get Instance method that acts as a constructor, returns the instance of the singleton object
ChainObjectHandler *ChainObjectHandler::getInstance() {
    if (instance == nullptr) {
        instance = new ChainObjectHandler();
    }
    return instance;
}

// Method for executing operations in the chain of objects
void ChainObjectHandler::ExecuteOperation(const Operations::op &operation) {

}

void ChainOperationHandler::addNodeToChain(IChainNode *newNode) {
    if (!head) {
        head = newNode;
    } else {
        IChainNode *current = head;
        while (current->nextNode != nullptr) {
            current = current->nextNode;
        }
        current->nextNode = newNode;
    }
}