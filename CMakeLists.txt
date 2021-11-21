cmake_minimum_required(VERSION 3.12)

############################################################################################
# CONDITIONAL FLAG (to run simulation on CPU or GPU)
# 
# For GPU:                      For CPU:
# set(ENABLE_CUDA YES)       set(ENABLE_CUDA NO)                       
#
# "YES"/GPU choice only available if CUDA library is installed and the GPU is CUDA capable.
############################################################################################
set(ENABLE_CUDA YES)

# Creates the Graphitti project with the correct languages, depending on if using GPU or not
# If using CUDA, also verify the CUDA package and set required CUDA variables
if(ENABLE_CUDA)	
        message("\n----Generating Makefile for Graphitti GPU version----")
        project(Graphitti LANGUAGES CXX CUDA C)
        # Verify CUDA package is present
        find_Package(CUDA REQUIRED)
        # Set the USE_GPU preproccesser macro so that GPU code will be compiled.
        add_compile_definitions(USE_GPU)
        # Specify the CUDA architecture/gencode that will be targeted
        ### Set gencode and architechture variables to the correct values for your specific NVIDIA hardware
        set(CMAKE_CUDA_ARCHITECTURES 37)        
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_37,code=sm_37)    
else()
        message("\n----Generating Makefile for Graphitti CPU version----")
        project(Graphitti LANGUAGES CXX C)
endif()

# Setting the base version to C++ 11
set(CMAKE_CXX_STANDARD 11)

# set(DEBUG_MODE YES) for debugging, no optimization
# set(DEBUG_MODE NO) for production code, -O3 optimization enabled
set(DEBUG_MODE NO)

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

# Collect source files and create libraries
# Libraries containing CUDA .cu files should be created as STATIC to ensure proper linking

# Create Connections library
file(GLOB_RECURSE Connections_Source Simulator/Connections/*.cpp Simulator/Connections/*.h Simulator/Connections/*.cu)
add_library(Connections STATIC ${Connections_Source})

# Create Edges library
file(GLOB_RECURSE Edges_Source Simulator/Edges/*.cpp Simulator/Edges/*.h Simulator/Edges/*.cu)
add_library(Edges STATIC ${Edges_Source})

# Create Layouts Library
file(GLOB_RECURSE Layouts_Source Simulator/Layouts/*.cpp Simulator/Layouts/*.h)
add_library(Layouts ${Layouts_Source})

# Create Vertices library
file(GLOB_RECURSE Vertices_Source Simulator/Vertices/*.cpp Simulator/Vertices/*.h Simulator/Vertices/*.cu)
add_library(Vertices STATIC ${Vertices_Source})
# The Vertices library has a dependency on the Edges library
target_link_libraries(Vertices Edges)

# Create FunctionNodes library
file(GLOB FunctionNodes_Source Simulator/Core/FunctionNodes/*.cpp Simulator/Core/FunctionNodes/*.h)
add_library(FunctionNodes ${FunctionNodes_Source})

# Create Core library
file(GLOB Core_Source Simulator/Core/*.cpp Simulator/Core/*.h Simulator/Core/*.cu)
# Remove Driver from the list of files, don't want 'main' in the library
list(REMOVE_ITEM Core_Source "${CMAKE_CURRENT_SOURCE_DIR}/Driver.cpp")
add_library(Core STATIC ${Core_Source})

# Create Recorders library
file(GLOB_RECURSE Recorders_Source Simulator/Recorders/*.cpp Simulator/Recorders/*.h)
add_library(Recorders ${Recorders_Source})

# Create Matrix library
file(GLOB Matrix_Source Simulator/Utils/Matrix/*.cpp Simulator/Utils/*.h)
# Remove these files from the library as they won't compile
list(REMOVE_ITEM Matrix_Source "${CMAKE_CURRENT_SOURCE_DIR}/Simulator/Utils/Matrix/MatrixFactory.h")
list(REMOVE_ITEM Matrix_Source "${CMAKE_CURRENT_SOURCE_DIR}/Simulator/Utils/Matrix/MatrixFactory.cpp")
add_library(Matrix ${Matrix_Source})

# Create RNG library
file(GLOB RNG_Source Simulator/Utils/RNG/*.cpp Simulator/Utils/RNG/*.h Simulator/Utils/RNG/*.cu)
# Remove demo from file list as it contains a main and it will cause compilation errors
list(REMOVE_ITEM RNG_Source "${CMAKE_CURRENT_SOURCE_DIR}/Simulator/Utils/RNG/MersenneTwister_demo.cu")
add_library(RNG STATIC ${RNG_Source})

# Create Utils library
file(GLOB Utils_Source  Simulator/Utils/*.cpp Simulator/Utils/*.h)
add_library(Utils ${Utils_Source})

# Used to locate and run other CMakeLists.txt files from Third Party resources for further compilation of the project.
add_subdirectory(ThirdParty)

# Googletest subdirectories support
add_subdirectory(Testing/lib/googletest-master)
include_directories(Testing/lib/GoogleTest/googletest-master/googletest/include)

# Set CUDA_SEPERABLE_COMPILATION to ON for all libraries that contain .cu CUDA files
if(ENABLE_CUDA)
        # Enables CUDA code in these libraries to be compiled into seperate object files and then linked together
        set_property(TARGET Edges PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET Vertices PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET Core PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET Connections PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET RNG PROPERTY CUDA_SEPARABLE_COMPILATION ON)
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
        # Add the ggraphitti GPU executable
        add_executable(ggraphitti Simulator/Core/Driver.cpp)
        # Link the combined library into the 'graphitti' executable.
	target_link_libraries(ggraphitti combinedLib)
else()
        # Add the cgraphitti CPU executable
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
