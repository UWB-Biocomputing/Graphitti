cmake_minimum_required(VERSION 3.12)

project(Graphitti LANGUAGES CUDA CXX C)

# Setting the base version to C++ 11
set(CMAKE_CXX_STANDARD 11)

############################################################################################
# CONDITIONAL FLAG (to run simulation on CPU or GPU)
# 
# For GPU:                      For CPU:
# set(ENABLE_CUDA YES)       set(ENABLE_CUDA NO)                       
#
# "YES"/GPU choice only available if CUDA library is installed and the GPU is CUDA capable.
############################################################################################
set(ENABLE_CUDA YES)

# set(DEBUG_MODE YES) for debugging, no optimization
# set(DEBUG_MODE NO) for production code, -O3 optimization enabled
set(DEBUG_MODE NO)

# Enables the CUDA language and locates the NVCC compiler. 
# Sets the USE_GPU preproccesser macro so that GPU code will be compiled.
#
# ENABLE_CUDA also compiles each library that contains CUDA with the following line...
# set_property(TARGET target PROPERTY CUDA_ARCHITECTURES 35-virtual) 
# so the compiler knows the CUDA architecture being used is the SM_35
if(ENABLE_CUDA)	
        find_Package(CUDA REQUIRED)        
        SET(CMAKE_LINKER CUDA)
        message("\n----Generating Makefile for Graphitti GPU version----")
        include_directories("${CUDA_INCLUDE_DIRS}")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -lineinfo; -std=c++11; -expt-extended-lambda; -O3; -use_fast_math; -rdc=true;)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_35,code=sm_35)
        string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_35,code=sm_35")
        add_compile_definitions(USE_GPU)
else()
        message("\n----Generating Makefile for Graphitti CPU version----")
endif()

# HDF5 Support, finds HDF5 package for C and C++ and links the hdf5 libaries to the executable later in the file.
find_package(HDF5 COMPONENTS C CXX)
if (HDF5_FOUND)
        message("-- HDF5 version ${HDF5_VERSION} located. HDF5 recorders are available.")
        link_directories( ${HDF5_LIBRARY_DIRS} )
        include_directories( ${HDF5_INCLUDE_DIRS} )
        add_compile_definitions(HDF5)
else()
        message("-- HDF5 library was not located. Please only use XML recorders.")
endif()

# Get git commit ID
# Change to "git rev-parse --short HEAD" for short commit ID
execute_process(
        COMMAND
        git rev-parse HEAD
        OUTPUT_VARIABLE        GIT_COMMIT_ID )

# GIT_COMMIT_ID has trailing whitespaces
string(REGEX REPLACE "\n$" "" GIT_COMMIT_ID "${GIT_COMMIT_ID}")

# Save GIT_COMMIT_ID to config.h, which is included in Driver.cpp
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
                ${CMAKE_BINARY_DIR}/config.h
                @ONLY )

set(CMAKE_INCLUDE_CURRENT_DIR ON)

#set(CMAKE_VERBOSE_MAKEFILE TRUE)

# Setting the location of the executable to be in the top-level directory. This helps when using file paths during
# runtime.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Set extra warning flags
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

if (NOT DEBUG_MODE)
        message("-- Setting Optimization flag: O3")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

#define TIXML_USE_STL as a preproccersser macro to use the C++ standard library with TinyXML
add_compile_definitions(TIXML_USE_STL)
message("-- Setting Compile Definition: TIMXL_USE_STL")

# Used to define file paths for #include definitions. For example you can write:
# #include "Simulator.h"
# rather than:
# #include "Simulator/Core/Simulator.h"
# if the file path "Simulator/Core" is in the following statement.
include_directories(
        ../Graphitti
        Simulator
        Simulator/Connections
        Simulator/Connections/Neuro
        Simulator/Connections/NG911
        Simulator/Core
        Simulator/Core/FunctionNodes
        Simulator/Edges
        Simulator/Edges/Neuro
        Simulator/Edges/NG911
        Simulator/Layouts
        Simulator/Layouts/Neuro
        Simulator/Layouts/NG911
        Simulator/Recorders
        Simulator/Recorders/Neuro
        Simulator/Recorders/NG911
        Simulator/Vertices
        Simulator/Vertices/Neuro
        Simulator/Vertices/NG911
        Simulator/Utils
        Simulator/Utils/Matrix
        Simulator/Utils/RNG
        Testing
        ThirdParty
        ThirdParty/cereal
        ThirdParty/TinyXPath
        ThirdParty/log4cplus-2.0.2/include)

# Used to locate and run other CMakeLists.txt files for further compilation of the project.
add_subdirectory(Simulator)
add_subdirectory(ThirdParty)

# Googletest subdirectories support
add_subdirectory(Testing/lib/googletest-master)
include_directories(Testing/lib/GoogleTest/googletest-master/googletest/include)

if(ENABLE_CUDA)
        set_property(TARGET Edges PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET Vertices PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET Core PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET Connections PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        target_compile_options(Edges PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:  -g; -arch=sm_35; -rdc=true;>)
        target_compile_options(Vertices PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:  -g; -arch=sm_35; -rdc=true;>)
        target_compile_options(Core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:  -g; -arch=sm_35; -rdc=true;>)
        target_compile_options(Connections PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:  -g; -arch=sm_35; -rdc=true;>)
        target_compile_options(Utils PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:  -g; -arch=sm_35; -rdc=true;>)
endif()

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

# ------ SIMULATOR EXECUTABLE -------
# Add all files that aren't in a library and are needed to run
if(ENABLE_CUDA)
        # Set CUDA and linker properties for combinedLib
        set_target_properties(combinedLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET combinedLib PROPERTY LINKER_LANGUAGE CUDA)
        set_property(TARGET combinedLib PROPERTY CUDA_ARCHITECTURES 35)

        # Add the Graphitti GPU Executable
        add_executable(ggraphitti Simulator/Core/Driver.cpp)
        
        # Link CUDA Libraries to executable and set CUDA Properties
        target_link_libraries(ggraphitti ${CUDA_LIBRARIES})
        set_target_properties(ggraphitti PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET ggraphitti PROPERTY LANGUAGE CUDA)
        set_property(TARGET ggraphitti PROPERTY CUDA_ARCHITECTURES 35)
        set_property(TARGET ggraphitti PROPERTY LINKER_LANGUAGE CUDA)

        # Link the combined library into the 'graphitti' executable.
	target_link_libraries(ggraphitti combinedLib)
else()
	add_executable(cgraphitti Simulator/Core/Driver.cpp)
	# Link the combined library into the 'graphitti' executable.
	target_link_libraries(cgraphitti combinedLib)
endif()
# ------ TESTS EXECUTABLE ------
# Add the file that contains main (RunTests.cpp) and all test files. GoogleTest will only recognize them if they are
# included in the executable.
add_executable(tests
        Testing/RunTests.cpp
        Testing/Core/OperationManagerTests.cpp
        Testing/Core/EdgeIndexMapTests.cpp
        Testing/Core/FunctionNodeTests.cpp
        Testing/Core/SimulatorTests.cpp
        Testing/Core/OperationManagerTestingClass.h
        Testing/Utils/ParameterManagerTests.cpp)

# Links the Googletest framework with the testing executable
target_link_libraries(tests gtest gtest_main)

# Link the combined library into the 'tests' executable.
target_link_libraries(tests combinedLib)
