#include <iostream>

#include "gtest/gtest.h"

using namespace std;

int main(int argc, char *argv[]) {
    testing::InitGoogleTest();
    int result = RUN_ALL_TESTS();
    if (result != 0) {
        cerr << "\nError occured while running the tests. Error number: " << result << endl;
    }
    return 0;
}