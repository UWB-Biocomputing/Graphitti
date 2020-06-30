//
// Created by chris on 6/23/2020.
//

#ifndef SUMMEROFBRAIN_Edges_H
#define SUMMEROFBRAIN_Edges_H

#include "IChainNode.h"

using namespace std;

class Edges : public IChainNode {
public:
    Edges();

    IChainNode *SetNextNode(IChainNode *nextNode);

    void PerformOperation(const Operations::op &operation);
};


#endif //SUMMEROFBRAIN_Edges_H
