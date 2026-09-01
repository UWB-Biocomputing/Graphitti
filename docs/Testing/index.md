# 3. Testing

Information on test config files for regression testing, and testing that has been done internally regarding potential improvements to Graphitti.

## 3.1 Unit Tests

We use [GoogleTest](../Developer/GoogleTestsTutorial.md) to develop our unit tests. 

To integrate your unit tests using GoogleTest in Graphitti you can follow these steps:
1. Open the CMakeLists.txt file in the root directory of Graphitti
2. Locate at the bottom of the file where the `tests` executable is defined and add your test file to the list of source files.
3. Build and run your tests using the Graphitti build system and use `./tests` to run the unit tests.

Please note that Graphitti follows the [singleton design pattern], and several of its classes, such as Simulator, ParameterManager, OperationManager, and GraphManager, are implemented as singletons. If your test scenario requires the instantiation of these classes, it may be necessary to create a separate executable specifically for your tests.

By creating a separate executable, you can ensure that the singleton instances used in the test environment are isolated from the main application's singleton instances. This approach helps maintain the desired behavior and avoid segmentation fault errors.

To create a separate executable for your test case in Graphitti, follow these steps:

1. Open the CMakeLists.txt file in the root directory of Graphitti.
2. Scroll to the bottom of the file and add the following code to create a new executable for your test case:

```
add_executable(YOUR_EXECUTABLE_TEST_NAME
        Testing/RunTests.cpp
        Testing/UnitTesting/YOUR_TEST_FILE.cpp)

# Link the necessary libraries and frameworks
target_link_libraries(YOUR_EXECUTABLE_TEST_NAME gtest gtest_main)
target_link_libraries(YOUR_EXECUTABLE_TEST_NAME combinedLib)
```
Make sure to replace "YOUR_EXECUTABLE_TEST_NAME" with the desired name for your test executable. Also, update the paths to the testing files according to your project structure.

3. After adding the code, save the CMakeLists.txt file.

4. Additionally, open the .gitignore file in the root directory of your project and add YOUR_EXECUTABLE_TEST_NAME to ignore the test executable.

5.  Build and run your tests using the Graphitti build system and use `./YOUR_EXECUTABLE_TEST_NAME` to run the unit tests.

## 3.2 Array Performance Testing

Testing the efficiency of C++ arrays, Vectors, and Valarrays.

- [Code](ArrayPerformance/ArraySpeedTest.cpp)
- [Writeup](ArrayPerformance/ArrayPerformance.md)

## 3.2 Dynamic Cast Performance Testing

Testing the performance impact of many dynamic_cast conversions.

- [Code](CastingTest/CastingTest.cpp)
- [Writeup](CastingTest/CastingTest.md)

## 3.3 Test Config Files

Documentation of the changing and constant parameters in a set of 12 test config files.

- [Writeup](TestConfigFileParameters/testConfigFileParameters.md)

### Running The Unit And Regression tests
Unit and regression tests are important to minimize the chances that changes to the code base will produce undesired side effects. For specific performance testing, clear cache by running `make clean` before compiling. 

We have a battery of tests that are run by a GitHub action on any `push` or `pull request` against the master branch. These tests are only executed for the CPU implementation of Graphitti.

The same tests executed by the described GitHub action can be run locally with the `RunTests.sh` bash script inside the `Testing` directory. The script can be told to exercise the tests against the GPU implementation by running it with the `-g` flag.

To run the tests against the CPU implementation, inside the `Testing` directory run:

    bash RunTests.sh

To run the tests against the GPU implementation, inside the `Testing` directory run:

    bash RunTests.sh -g

**Note**: Currently, the GPU regression tests fail because the random numbers generated are different from the ones
generated during the CPU execution, causing the result files to be different to the CPU known good results.

---------
[<< Go back to the Graphitti home page](../index.md)

[//]: # (Moving URL links to the bottom of the document for ease of updating - LS)
[//]: # (Links to repo items which exist outside of the docs folder need an absolute link.)

[singleton design pattern]: <https://en.wikipedia.org/wiki/Singleton_pattern>