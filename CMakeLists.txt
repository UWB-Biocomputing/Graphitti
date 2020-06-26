cmake_minimum_required(VERSION 3.10)
project(SummerOfBrain)

set(CMAKE_CXX_STANDARD 14)

# Add all directories that contain used classes
include_directories(ChainOfResponsibility)

# Including Googletest subdirectories
add_subdirectory(Testing/lib/googletest-master)
include_directories(Testing/lib/googletest-master/googletest/include)

# Add all class paths needed for main to be ran
add_executable(SummerOfBrain
        Testing/ChainNodeTest.cpp
        ChainOfResponsibility/Dog.cpp
        ChainOfResponsibility/Dog.h
        ChainOfResponsibility/Fish.cpp
        ChainOfResponsibility/Fish.h
        ChainOfResponsibility/IChainNode.h)

# Links the Googletest framework with Braingrid
target_link_libraries(SummerOfBrain gtest gtest_main)
