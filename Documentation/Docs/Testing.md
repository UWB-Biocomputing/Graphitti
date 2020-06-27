


## Googletest Tutorial

Googletest is a c++ testing framework that supports system, integration and unit testing. It allows users to make independent and repeatable tests to assure that changes don’t alter the functionality of the system.

The following link points to the documentation of Googletest which includes valuable information on what features it offers and a basic tutorial on how to write tests.

[https://github.com/google/googletest/blob/master/googletest/docs/primer.md](https://github.com/google/googletest/blob/master/googletest/docs/primer.md)

  

The following link is a helpful video describing how to write tests and test structs using Googletest.

[https://www.youtube.com/watch?v=16FI1-d2P4E&](https://www.youtube.com/watch?v=16FI1-d2P4E&)

  

In Braingrid we use this framework to write unit tests for each object class. Follow these steps when making a new object class in the project.

  

1.  Create a new .cpp class in the testing folder
    
2.  Name the .cpp file <class-name>Tests.cpp
    
3.  Write “#include “gtest/gtest.h” and any other .h files in the dependencies of the .cpp test class
    
4.  Write tests that assure the class is working properly
    

1.  Name the tests after the function it’s testing and what values are being provided
    

6.  In main, make sure that RUN_ALL_TESTS() is being called
    
7.  Update CMakeLists.txt file in top-level folder
    

1.  Add all new class paths to the CMakeLists.txt in the add_executable(...) parameters
    
2.  Add all new directories to the CMakeLists.txt in the include_directories(...) parameters
    

9.  Build and run to run all test cases
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE4ODUwMjAyMjFdfQ==
-->