//
// Created by chris on 6/30/2020.
//

#include "ChainNode.h"

ChainNode::ChainNode(std::function<void()> function) {
    this->function = function;
}

void ChainNode::performOperation() {
    __invoke(function);
}