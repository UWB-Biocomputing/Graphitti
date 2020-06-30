cmake_minimum_required(VERSION 3.10)
project(SummerOfBrain)

set(CMAKE_CXX_STANDARD 14)

# Add all directories that contain used classes
include_directories(ChainOfResponsibility
        Testing)

# Including Googletest subdirectories
add_subdirectory(Testing/lib/googletest-master)
include_directories(Testing/lib/googletest-master/googletest/include)

# Add all class paths needed for main to be ran
add_executable(SummerOfBrain
        Testing/RunTests.cpp
        ChainOfResponsibility/Foo.h
        ChainOfResponsibility/Foo.cpp
        ChainOfResponsibility/ChainNode.h
        ChainOfResponsibility/ChainNode.cpp
        ChainOfResponsibility/ChainOperationManager.h
        ChainOfResponsibility/ChainOperationManager.cpp
        ChainOfResponsibility/Operations.h
        Testing/ChainOfResponsibility/ChainNodeTests.cpp
        Testing/ChainOfResponsibility/ChainOperationManagerTests.cpp)

# Links the Googletest framework with Braingrid
target_link_libraries(SummerOfBrain gtest gtest_main)
