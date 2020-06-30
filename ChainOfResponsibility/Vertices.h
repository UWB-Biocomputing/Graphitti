//
// Created by chris on 6/22/2020.
//

#pragma once

#include "IChainNode.h"

class Vertices : public IChainNode {
public:
    Vertices();

    IChainNode *SetNextNode(IChainNode *nextNode);

    void PerformOperation(const Operations::op &operation);

private:
    void allocateMemory();

};

