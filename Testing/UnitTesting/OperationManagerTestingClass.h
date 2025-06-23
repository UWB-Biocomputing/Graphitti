/**
 * @file OperationManagerTestingClass.h
 *
 * @brief  Sample class used for testing purposes. 
 * Will be replaced once lower level classes are imported into the project.
 *
 * @ingroup Testing/UnitTesting
 */

#pragma once

#include "OperationManager.h"
#include <iostream>

using namespace std;

class IFoo {
public:
   virtual void loadParameters() = 0;

   virtual void printParameters() = 0;

   virtual void loadEpochInputs(uint64_t currentStep, uint64_t endStep) = 0;
};


class Foo : public IFoo {
public:
   Foo()
   {
   }

   virtual void loadParameters()
   {
      cout << "Foo loading parameters" << endl;
   }

   virtual void printParameters()
   {
      cout << "Foo printing parameters" << endl;
   }

   virtual void loadEpochInputs(uint64_t currentStep, uint64_t endStep)
   {
      cout << "Foo current step: " << currentStep << endl;
      cout << "Foo end step: " << endStep << endl;
   }
};


class Bar : public Foo {
public:
   Bar() : Foo()
   {
   }

   virtual void loadParameters()
   {
      cout << "Bar loading Parameters" << endl;
   }
};


class Car : public Bar {
public:
   Car() : Bar()
   {
   }
};
