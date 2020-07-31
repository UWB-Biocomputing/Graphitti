cmake_minimum_required(VERSION 3.10)
project(SummerOfBrain)

# Setting the base version to C++ 14
set(CMAKE_CXX_STANDARD 14)

# Set extra warning flags
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Used to define file paths for #include paths. For example you can write:
# #include "Simulator.h"
# rather than:
# #include "Simulation/Core/Simulator.h"
# if the file path is in the following statement.
include_directories(
        ../SummerOfBrain
        Simulation
        Simulation/Connections
        Simulation/Core
        Simulation/Core/FunctionNodes
        Simulation/Edges
        Simulation/Layouts
        Simulation/Recorders
        Simulation/Vertices
        Testing
        ThirdParty
        ThirdParty/cereal
        Utils
        Utils/RNG
        Utils/TinyXPath)

# Used to locate other CMakeLists.txt files for further compilation of the project.
add_subdirectory(Simulation)
add_subdirectory(Utils)

# Including Googletest subdirectories
add_subdirectory(Testing/lib/googletest-master)
include_directories(Testing/lib/GoogleTest/googletest-master/googletest/include)

# ------ CPU SIMULATOR EXECUTABLE -------
# Add all files that aren't in a library and are needed to run
#add_executable(BrainGrid)

# ------ GPU SIMULATOR EXECUTABLE ------
# Add all files that aren't in a library and are needed to run
#add_executable(BrainGrid-cuda)

# ------ TESTS EXECUTABLE ------
# Add the file that contains main (RunTests.cpp) and all test files. GoogleTest will only recognize them if they are
# included in the executable.
add_executable(Tests
        Testing/RunTests.cpp
        Testing/Core/Foo.h
        Testing/Core/Foo.cpp
        Testing/Core/FunctionNodeTests.cpp
        Testing/Core/OperationManagerTests.cpp
        Testing/Core/SynapseIndexMapTests.cpp
        Testing/Core/SimulatorTests.cpp)

# Links the Googletest framework with the testing executable
target_link_libraries(Tests gtest gtest_main)

# Link all libraries created in the project's subdirectories to the 'Tests' executable.
target_link_libraries(Tests
        Connections
        Core
        Layouts
        Edges
        Vertices
        FunctionNodes
        Utils
        RNG
        Matrix
        Inputs)