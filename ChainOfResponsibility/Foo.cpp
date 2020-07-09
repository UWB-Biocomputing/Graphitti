//
// Created by chris on 6/29/2020.
//

#include "Foo.h"
#include "Operations.h"
#include "OperationManager.h"
#include <functional>

using namespace std;

Foo::Foo() {
    auto function = std::bind(&Foo::allocateMemory, this);
    OperationManager::getInstance()->registerOperation(Operations::op::allocateMemory, function);
}

void Foo::allocateMemory() {
    //cout << "allocating memory" << endl;
}

void Foo::deallocateMemory() {
    //cout << "deallocating memory" << endl;
}