//
// Created by chris on 6/22/2020.
//

#ifndef SUMMEROFBRAIN_DOG_H
#define SUMMEROFBRAIN_DOG_H

#include "IChainNode.h"

class Dog : public IChainNode {
public:
    Dog();

    IChainNode *SetNextNode(IChainNode *nextNode);

    string PerformOperation(string request);

private:
    IChainNode *nextNode = NULL;
};


#endif //SUMMEROFBRAIN_DOG_H
