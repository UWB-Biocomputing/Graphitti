cmake_minimum_required(VERSION 3.10)
project(SummerOfBrain)

set(CMAKE_CXX_STANDARD 14)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Add all directories that contain required files
include_directories(
        ../SummerOfBrain
        Core
        Core/FunctionNodes
        Testing
        Utils
        Utils/TinyXPath)

# Googletest subdirectories support
add_subdirectory(Testing/lib/googletest-master)
include_directories(Testing/lib/googletest-master/googletest/include)

# Boost Library support
find_package(Boost COMPONENTS regex)

# TinyXPath support
add_subdirectory(ThirdParty/TinyXPath)

#target_include_directories(TinyXLib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Utils/TinyXPath")

# ------ CPU SIMULATOR EXECUTABLE -------
# Add all files that needed to build
#add_executable(BrainGrid)

# ------ GPU SIMULATOR EXECUTABLE ------
# Add all files that needed to build
#add_executable(BrainGrid-cuda)

# ------ TESTS EXECUTABLE ------
# Add all user created files that are need to build
add_executable(Tests
        Core/OperationManager.h
        Core/OperationManager.cpp
        Core/Operations.h
        Core/SynapseIndexMap.h
        Core/FunctionNodes/GenericFunctionNode.h
        Core/FunctionNodes/GenericFunctionNode.cpp
        Utils/ParameterManager.h
        Utils/ParameterManager.cpp
        Testing/RunTests.cpp
        Testing/Core/Foo.h
        Testing/Core/Foo.cpp
        Testing/Core/FunctionNodeTests.cpp
        Testing/Core/OperationManagerTests.cpp
        Testing/Core/SynapseIndexMapTests.cpp
        Testing/Utils/ParamaterManagerTests.cpp
        Utils/BGTypes.h)

# Links the required Boost libraries with the testing executable
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(Tests ${Boost_LIBRARIES})
endif()

# Links the Googletest framework with the testing executable
target_link_libraries(Tests gtest gtest_main)

# Links the TinyXPath library with the the testing executable
target_link_libraries(Tests TinyXLib)
