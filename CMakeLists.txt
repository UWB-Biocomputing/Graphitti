cmake_minimum_required(VERSION 3.10)
project(SummerOfBrain)

set(CMAKE_CXX_STANDARD 14)

include_directories(ChainOfResponsibility)

add_subdirectory(Testing/lib/googletest-master)

include_directories(Testing/lib/googletest-master/googletest/include)

add_executable(SummerOfBrain
        ChainOfResponsibility/ChainNodeTest.cpp
        ChainOfResponsibility/Dog.cpp
        ChainOfResponsibility/Dog.h
        ChainOfResponsibility/Fish.cpp
        ChainOfResponsibility/Fish.h
        ChainOfResponsibility/IChainNode.h)

target_link_libraries(SummerOfBrain gtest gtest_main)
