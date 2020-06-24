//
// Created by chris on 6/22/2020.
//

#ifndef SUMMEROFBRAIN_ICHAINNODE_H
#define SUMMEROFBRAIN_ICHAINNODE_H

#include <string>

using namespace std;

class IChainNode {
public:
    virtual IChainNode *SetNextNode(IChainNode *nextNode) = 0;

    virtual string PerformOperation(string request) = 0;
};


#endif //SUMMEROFBRAIN_ICHAINNODE_H
