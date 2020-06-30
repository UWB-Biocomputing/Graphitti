//
// Created by chris on 6/29/2020.
//

#pragma once

#include <iostream>

using namespace std;

class IChainNode {
public:
    virtual void performOperation() = 0;
};
