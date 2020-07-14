#include "Foo.h"

#include <functional>

#include "Operations.h"
#include "OperationManager.h"

/*
 * Sample class used for testing purposes. Will be replaced once lower level classes are imported into the project.
 * Delete by the end of Summer 2020.
 */

using namespace std;

Foo::Foo() {
    auto function = std::bind(&Foo::allocateMemory, this);
    OperationManager::getInstance().registerOperation(Operations::op::allocateMemory, function);
}

void Foo::allocateMemory() {
    //cout << "allocating memory" << endl;
}

void Foo::deallocateMemory() {
    //cout << "deallocating memory" << endl;
}