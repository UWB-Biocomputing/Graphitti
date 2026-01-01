# Unit Testing

Information on unit tests, test config files for regression testing, and testing that has been done internally regarding potential improvements to Graphitti.

## Unit Tests

We use [Googletest](GoogleTestsTutorial.md) to develop our unit tests.


### Running The Unit And Regression tests
Unit and regression tests are important to minimize the chances that changes to the code base will produce undesired side effects. 

We have a battery of tests that are run by a GitHub action on any `push` or `pull request` against the master branch. These tests are only executed for the CPU implementation of Graphitti.

The same tests executed by the described GitHub action can be run locally with the `RunTests.sh` bash script inside the `Testing` directory. The script can be told to exercise the tests against the GPU implementation by running it with the `-g` flag.

To run the tests against the CPU implementation, inside the `Testing` directory run:

    bash RunTests.sh

To run the tests against the GPU implementation, inside the `Testing` directory run:

    bash RunTests.sh -g

**Note**: Currently, the GPU regresssion tests fail because the random numbers generated are different from the ones
generated during the CPU execution, causing the result files to be different to the CPU known good results.

---------
[<< Go back to the Graphitti home page](../index.md)
