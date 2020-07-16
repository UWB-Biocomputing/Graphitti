cmake_minimum_required(VERSION 3.10)
project(SummerOfBrain)

set(CMAKE_CXX_STANDARD 14)

# Add all directories that contain used classes
include_directories(
        ../SummerOfBrain
        Core
        Core/FunctionNodes
        Testing)

# Including Googletest subdirectories
add_subdirectory(Testing/lib/googletest-master)
include_directories(Testing/lib/googletest-master/googletest/include)

# CPU SIMULATOR EXECUTABLE
# Add all class paths needed for dependencies
#add_executable(BrainGrid)

# GPU SIMULATOR EXECUTABLE
# Add all class paths needed for dependencies
#add_executable(BrainGrid-cuda)

# TEST EXECUTABLE
# Add all class paths needed for dependencies
add_executable(Tests
        Testing/RunTests.cpp
        Testing/Core/Foo.h
        Testing/Core/Foo.cpp
        Core/FunctionNodes/GenericFunctionNode.h
        Core/FunctionNodes/GenericFunctionNode.cpp
        Core/OperationManager.h
        Core/OperationManager.cpp
        Core/Operations.h
        Testing/Core/FunctionNodeTests.cpp
        Testing/Core/OperationManagerTests.cpp)

# Links the Googletest framework with the testing classes
target_link_libraries(Tests gtest gtest_main)
