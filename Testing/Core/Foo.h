/*
 * Sample class used for testing purposes. Will be replaced once lower level classes are imported into the project.
 * Delete by the end of Summer 2020.
 */

#pragma once

#include <iostream>

#include "OperationManager.h"

using namespace std;

class IFoo {
public:
   virtual void allocateMemory() = 0;

   virtual void deallocateMemory() = 0;
};


class Foo : public IFoo {
public:
    Foo() {
       auto function = std::bind(&Foo::allocateMemory, this);
       OperationManager::getInstance().registerOperation(Operations::op::allocateMemory, function);
    }

    virtual void allocateMemory() {
       cout << "Foo allocating memory" << endl;
    }

    virtual void deallocateMemory() {
       cout << "Foo deallocating memory" << endl;
    }
};


class Bar : public Foo {
public:
   Bar() : Foo() {
      auto function = std::bind(&Bar::allocateMemory, this);
      OperationManager::getInstance().registerOperation(Operations::op::allocateMemory, function);
   }

   virtual void allocateMemory() {
      cout << "Bar allocating memory" << endl;
   }
};


class Car : public Bar {
public:
   Car() : Bar() {

   }
};
