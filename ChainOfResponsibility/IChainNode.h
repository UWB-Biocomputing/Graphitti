#pragma once      

#include <string>

using namespace std;

// Interface
class IChainNode {
public:
    IChainNode *nextNode;
    
    // Method for setting the next node in the chain
    virtual IChainNode *SetNextNode(IChainNode *nextNode) = 0;

    // Generic operation
    virtual void PerformOperation(const Operations::op &operation) = 0;
};
