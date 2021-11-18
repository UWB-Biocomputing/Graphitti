# Googletest Tutorial

### Helpful Sources

Googletest is a C++ testing framework that supports system, integration and unit testing. It allows users to make independent and repeatable tests to assure that changes don’t alter the functionality of the system. 
	
The following link points to the documentation of Googletest which includes valuable information on what features it offers and a basic tutorial on how to write tests.
[http://google.github.io/googletest/primer.html](http://google.github.io/googletest/primer.html)

The following link is a helpful video describing how to write tests and test structs using Googletest.
[https://www.youtube.com/watch?v=16FI1-d2P4E&amp;](https://www.youtube.com/watch?v=16FI1-d2P4E&amp;)

#### Watch here:

[![video of unit testing](http://img.youtube.com/vi/16FI1-d2P4E/0.jpg)](http://www.youtube.com/watch?v=16FI1-d2P4E "C++ Unit Testing With Google Test Tutorial")

### Test Format

Following is a short explanation on how the tests are written for this project. In basic `TEST` declaration you will need to provide two parameters which will be used to name and organize the tests. The first parameter is known as the test suite. The test suite is what is used to group tests together. For the test suite, use the name of the class that is being tested. The second parameter is the name of the test. Name the test accordingly based on what aspect of the class is being tested. 

In both parameters, use Pascal Casing.

    TEST(ClassNameThatIsBeingTested, FunctionalityBeingTested) {}

Example: 

    TEST(OperationManager, AddingSingleOperation)

The following test declaration requires a test struct object rather than a test suite. Test structs are used to set a state for the object you are testing without reusing code. 

    TEST_F(NameOfTestStruct, FunctionalityBeingTested) {}

Example: 

    TEST_F(OperationManagerTestObject, OperationExecutionSuccess) {}

GoogleTest provides `EXPECT`s to check outputs and assure everything is working as expected. They work just like assertions except that `ASSERT` and `EXPECT` handle failures differently. When an `EXPECT` doesn’t match the predicted result, the test will fail and will go on to the next test. When an `ASSERT` fails the program will end. We want to run all the tests each time regardless if a test fails, so use `EXPECT` checks.

For each test make sure that you have an `EXPECT` so that we know whether or not a class is functionally correct.  If no output can be tested, use the `EXPECT_NO_FATAL_ERROR()`.

### Creating Test Files and Writing Unit Tests

In Graphitti we use this framework to write unit tests for each object class. Follow these steps when making a new object class in the project.

 1. Create a new .cpp file in a directory inside the testing folder named identically as the directory where the class is located. For example if you we’re writing tests for a class  located in Core, you would save the .cpp file in Testing/Core. 
 2. Name the .cpp file `Tests.cpp` 
 3. Write “`#include “gtest/gtest.h`” and any other .h files in the dependencies of the .cpp test class. 
 4. Write tests that assure the class is working properly. Follow the existing format while writing tests.
 * Name the tests after the function it’s testing and what values are being provided. 
 5. In main in `RunAllTests.cpp`, make sure that `RUN_ALL_TESTS()` is being called.
 6. Update `CMakeLists.txt` file in top-level project folder
* Add all new class paths to the `CMakeLists.txt` in the `add_executable(Tests...)` parameters. The testing files only needed to be included in the ‘Tests’ executable. 
* Add all new directories to the `CMakeLists.txt` in the `include_directories(...)` parameters. This will add  Build and run the Tests executable to run all tests.

[http://google.github.io/googletest/primer.html](http://google.github.io/googletest/primer.html)
