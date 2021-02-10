cmake_minimum_required(VERSION 3.12)
project(SummerOfBrain)

# Setting the base version to C++ 11
set(CMAKE_CXX_STANDARD 11)

# Enable the use of GPU. Only available if CUDA library is installed the GPU is CUDA capable.
set(BRAINGRID_CUDA NO)

# Enables the CUDA langauge and locates the NVCC compiler. 
# Sets the USE_GPU preproccesser macro so that GPU code will be compiled.
#
# BRAINGRID_CUDA also compiles each library that contains CUDA with the following line...
# set_property(TARGET target PROPERTY CUDA_ARCHITECTURES 35-virtual) 
# so the compiler knows the CUDA architecture being used is the SM_35
if(BRAINGRID_CUDA)
        enable_language(CUDA)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-g;-arch=sm_35;-rdc=true")
        add_compile_definitions(USE_GPU)
endif()

# HDF5 Support, finds HDF5 package for C and C++ and links the hdf5 libaries to the executable later in the file.
find_package(HDF5 COMPONENTS C CXX)
if (HDF5_FOUND)
        link_directories( ${HDF5_LIBRARY_DIRS} )
        include_directories( ${HDF5_INCLUDE_DIRS} )
        add_compile_definitions(HDF5)
endif()

# Setting the location of the executable to be in the top-level directory. This helps when using file paths during
# runtime.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Set extra warning flags
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

#define TIXML_USE_STL as a preproccersser macro to use the C++ standard library with TinyXML
add_compile_definitions(TIXML_USE_STL)

# Used to define file paths for #include definitions. For example you can write:
# #include "Simulator.h"
# rather than:
# #include "Simulation/Core/Simulator.h"
# if the file path "Simulation/Core" is in the following statement.
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
        Simulation/Utils
        Testing
        ThirdParty
        ThirdParty/cereal
        ThirdParty/TinyXPath
        ThirdParty/log4cplus-2.0.2/include)

# Used to locate and run other CMakeLists.txt files for further compilation of the project.
add_subdirectory(Simulation)
add_subdirectory(ThirdParty)

# Googletest subdirectories support
add_subdirectory(Testing/lib/googletest-master)
include_directories(Testing/lib/GoogleTest/googletest-master/googletest/include)

# ------ SIMULATOR EXECUTABLE -------
# Add all files that aren't in a library and are needed to run
add_executable(braingrid Simulation/Core/BGDriver.cpp)

# ------ TESTS EXECUTABLE ------
# Add the file that contains main (RunTests.cpp) and all test files. GoogleTest will only recognize them if they are
# included in the executable.
add_executable(tests
        Testing/RunTests.cpp
        Testing/Core/OperationManagerTests.cpp
        Testing/Core/SynapseIndexMapTests.cpp
        Testing/Core/FunctionNodeTests.cpp
        Testing/Core/SimulatorTests.cpp
        Testing/Core/OperationManagerTestingClass.h
        Testing/Utils/ParamaterManagerTests.cpp)

# Links the Googletest framework with the testing executable
target_link_libraries(tests gtest gtest_main)

# Link all libraries created in the project's subdirectories into a combined library.
add_library(combinedLib INTERFACE)
target_link_libraries(combinedLib INTERFACE
        # Simulator
        Core
        Connections
        Edges
        Layouts
        Recorders
        Vertices
        FunctionNodes

        # Utils
        Utils
        RNG
        #Inputs *** inputs aren't in the project ***
        Matrix

        # ThirdParty
        TinyXPath
        log4cplus
        paramcontainer)

# Link HDF5 package libaries if to the combined library.
if (HDF5_FOUND)
        target_link_libraries(combinedLib INTERFACE ${HDF5_LIBRARIES})
endif()

# Link the combined library into the 'braingrid' executable.
target_link_libraries(braingrid combinedLib)

# Link the combined library into the 'tests' executable.
target_link_libraries(tests combinedLib)
