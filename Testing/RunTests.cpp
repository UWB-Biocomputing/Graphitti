//
// Created by Chris O'Keefe on 6/26/2020.
//

#include "gtest/gtest.h"
#include "OperationManager.h"
#include "Foo.h"
#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {
    testing::InitGoogleTest();
    int result = RUN_ALL_TESTS();
    if (result != 0) {
        cerr << "\nError occured while running the tests. Error number: " << result << endl;
    }
    return 0;
}