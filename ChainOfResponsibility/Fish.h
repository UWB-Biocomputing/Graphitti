//
// Created by chris on 6/23/2020.
//

#ifndef SUMMEROFBRAIN_FISH_H
#define SUMMEROFBRAIN_FISH_H

#include "IChainNode.h"

using namespace std;

class Fish : public IChainNode {
public:
    Fish();

    IChainNode *SetNextNode(IChainNode *nextNode);

    string PerformOperation(string request);

private:
    IChainNode *nextNode = NULL;
};


#endif //SUMMEROFBRAIN_FISH_H
