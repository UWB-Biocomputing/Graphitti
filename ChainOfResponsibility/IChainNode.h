//
// Created by chris on 6/22/2020.
//

#ifndef SUMMEROFBRAIN_ICHAINNODE_H
#define SUMMEROFBRAIN_ICHAINNODE_H

#include <string>

using namespace std;

// Interface
class IChainNode {
public:
    // Method for setting the next node in the chain
    virtual IChainNode *SetNextNode(IChainNode *nextNode) = 0;

    // Generic operation
    virtual string PerformOperation(string request) = 0;
};


#endif //SUMMEROFBRAIN_ICHAINNODE_H
