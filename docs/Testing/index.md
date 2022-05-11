## Testing

Information on unit tests, test config files for regression testing, and testing that has been done internally regarding potential improvements to Graphitti.

### Unit Tests
We use [Googletest](GoogleTestsTutorial.md) to develop our unit tests.

### Array Performance Testing
Testing the efficency of C++ arrays, Vectors, and Valarrays.

- [Code](ArrayPerformance/ArraySpeedTest.cpp)
- [Writeup](ArrayPerformance/ArrayPerformance.md)

### Dynamic Cast Performance Testing
Testing the performance impact of many dynamic_cast conversions.

- [Code](CastingTest/CastingTest.cpp)
- [Writeup](CastingTest/CastingTest.md)

### Test Config Files
Documentation of the changing and constant parameters in a set of 12 test config files.

- [Writeup](TestConfigFileParameters/testConfigFileParameters.md)

### Running The Unit And Regression tests
Unit and regression tests are important to minimize the chances that changes to the code base will produce undesired side effects. 

We have a battery of tests that are run by a GitHub action on any `push` or `pull request` against the master branch. These tests are only executed for the CPU implementation of Graphitti.

The same tests executed by the described GitHub action can be run locally with the `RunTests.sh` bash script inside the `Testing` directory. The script can be told to exercise the tests against the GPU implementation by running it with the `-g` flag.

To run the tests against the CPU implementation, inside the `Testing` directory run:

    bash RunTests.sh

To run the tests against the GPU implementation, inside the `Testing` directory run:

    bash RunTests.sh -g

---------
[<< Go back to the Graphitti home page](..)
