//
// Created by chris on 6/29/2020.
//

#ifndef SUMMEROFBRAIN_ICHAINNODE_H
#define SUMMEROFBRAIN_ICHAINNODE_H

class IChainNode {
public:
    virtual IChainNode *setNextNode(IChainNode *nextNode) = 0;

    virtual void performOperation() = 0;
};

#endif //SUMMEROFBRAIN_ICHAINNODE_H
