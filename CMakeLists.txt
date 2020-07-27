cmake_minimum_required(VERSION 3.10)
project(SummerOfBrain)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Add all directories that contain required classes
include_directories(
        ../SummerOfBrain
        Connections
        Core
        Core/FunctionNodes
        Testing
        Utils
        Utils/TinyXPath)

# Including Googletest subdirectories
add_subdirectory(Testing/lib/googletest-master)
include_directories(Testing/lib/GoogleTest/googletest-master/googletest/include)

# ------ CPU SIMULATOR EXECUTABLE -------
# Add all class paths needed for dependencies
#add_executable(BrainGrid)

# ------ GPU SIMULATOR EXECUTABLE ------
# Add all class paths needed for dependencies
#add_executable(BrainGrid-cuda)

# ------ TESTS EXECUTABLE ------
# Add all class paths needed for dependencies
add_executable(Tests
        Simulation/Core/OperationManager.h
        Simulation/Core/OperationManager.cpp
        Simulation/Core/Operations.h
        Simulation/Core/SynapseIndexMap.h
        Simulation/Core/FunctionNodes/GenericFunctionNode.h
        Simulation/Core/FunctionNodes/GenericFunctionNode.cpp
        Testing/RunTests.cpp
        Testing/Core/Foo.h
        Testing/Core/Foo.cpp
        Testing/Core/FunctionNodeTests.cpp
        Testing/Core/OperationManagerTests.cpp
        Testing/Core/SynapseIndexMapTests.cpp
        Utils/BGTypes.h)

# Links the Googletest framework with the testing executable
target_link_libraries(Tests gtest gtest_main)
