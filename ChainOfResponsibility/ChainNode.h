//
// Created by chris on 6/22/2020.
//

#pragma once

#include "Operations.h"
#include "IChainNode.h"
#include <functional>
#include <iostream>

using namespace std;

class ChainNode : public IChainNode {
public:
    ChainNode(std::function<void()> function);

    // Generic operation
    void performOperation() override;

private:
    std::function<void()> function;
};


