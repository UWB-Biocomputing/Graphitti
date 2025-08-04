cmake_minimum_required(VERSION 3.12)

############################################################################################
#CONDITIONAL FLAG(to run simulation on CPU or GPU)
#
#For GPU : For CPU:
#set(ENABLE_CUDA YES) set(ENABLE_CUDA NO)
#
#You can also pass this flag when running cmake from the command line like this:
#
#cmake -D ENABLE_CUDA=YES ..
#
#"YES" / GPU choice only available if CUDA library is installed and the GPU is CUDA capable.
############################################################################################
if(NOT ENABLE_CUDA)
        set(ENABLE_CUDA NO)
endif()

#CONDITIONAL FLAG to turn on the performance metrics 
if(NOT PERFORMANCE_METRICS)
        set(PERFORMANCE_METRICS NO)
endif()

############################################################################################
#CONDITIONAL FLAG to change target architecture for the GPU simulator from the default
#
#You can pass this flag when running cmake from the command line like this, setting TARGET_ARCH \
#       to your desired architecture:                                                            \
#
#cmake -D ENABLE_CUDA=YES -D TARGET_ARCH=70 ..
#
#"YES" / GPU choice only available if CUDA library is installed and the GPU is CUDA capable.
#If no TARGET_ARCH is passed in then it will default to 37 which is the kepler architecture
############################################################################################
if(NOT DEFINED TARGET_ARCH)
        set(TARGET_ARCH 37)
endif()

#CONDITIONAL FLAG to turn on the validation mode
if(NOT VALIDATION_MODE)
        set(VALIDATION_MODE NO)
endif()

#Creates the Graphitti project with the correct languages, depending on if using GPU or not
#If using CUDA, also verify the CUDA package and set the required CUDA variables
if(ENABLE_CUDA)	
        message("\n----Generating Makefile for Graphitti GPU version----")
        project(Graphitti LANGUAGES CXX CUDA C)
#Verify CUDA package is present
        find_Package(CUDA REQUIRED)
#Set the USE_GPU preprocessor macro so that GPU code will be compiled.
        add_compile_definitions(USE_GPU)
#Specify the CUDA architecture / gencode that will be targeted
        ### Set gencode and architecture variables to the correct values for your specific NVIDIA hardware
        set(CMAKE_CUDA_ARCHITECTURES ${TARGET_ARCH})
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_${TARGET_ARCH},code=sm_${TARGET_ARCH})        
        message(STATUS "Using CUDA architecture: ${TARGET_ARCH}")
            
else()
        message("\n----Generating Makefile for Graphitti CPU version----")
        project(Graphitti LANGUAGES CXX C)
endif()

# -----------------------------------------------------------------------------
# Build Type Configuration
#
# CMake support for different build types controling optimization, debugging and profiling:
#
#   - Debug         : No optimizations (`-O0`), includes debug symbols (`-g`).
#   - Release       : Optimized build (`-O3`), removes debug symbols.
#   - RelWithDebInfo: Optimized (`-O2`) but keeps debug symbols (`-g`) for profiling.
#   - Profiling     : Custom build type (defined in this project) that enables:
#                     - CPU profiling via `-pg` (GPROF)
#                     - CUDA profiling via `-lineinfo` (for Nsight Compute)
#
# Selecting a Build Type:
#   - By default, CMake does NOT set a build type for single-config generators.
#   - If no build type is specified, this script defaults to "Release" for performance.
#   - You can explicitly set the build type when configuring CMake:
#
#       cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug      # Debug mode
#       cmake -S . -B build -DCMAKE_BUILD_TYPE=Release    # Release mode
#       cmake -S . -B build -DCMAKE_BUILD_TYPE=Profiling  # Profiling mode 
#       
#       If you don't want to pass in the build type flag, you can edit this file and add...
#               set(CMAKE_BUILD_TYPE "Debug") or whichever build type you want
# -----------------------------------------------------------------------------
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;Profiling" CACHE STRING "Supported build types" FORCE)

# Ensure single-config generators use a valid default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type." FORCE)
endif()

# Set flags for all build types
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
# We should consider using the -DNDEBUG flag for release code, it disables assert() calls and is higher performance
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

# Define a custom build type: "Profiling"
set(CMAKE_CXX_FLAGS_PROFILING "-pg -O2")
set(CMAKE_EXE_LINKER_FLAGS_PROFILING "-pg")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

# Apply the correct flags based on the selected build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
    if(ENABLE_CUDA)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")   
    endif()
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
    if(ENABLE_CUDA)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    endif()
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Profiling")
    message(STATUS "Profiling build enabled: Adding -pg (GPROF)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_PROFILING}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS_PROFILING}")
    if(ENABLE_CUDA)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")   
#       set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -Xptxas=-v")   
    endif()
endif()


#   Gprof is a performance analysis tool for Unix applications)
#Steps to run Gprof
#Step 01 : set build configuration to Profiling  ...   -DCMAKE_BUILD_TYPE=Profiling
#Step 02 : Compile and run the simulation on CPU or GPU as usual
#Step 03 : Run the generated gmon.out file from the build directory and save the output in an txt  \
#      file to improve readability                                                                  \
#If using CPU - "~/Graphitti/build$ gprof cgraphitti gmon.out > analysis_test.txt"
#If using GPU - "~/Graphitti/build$ gprof ggraphitti gmon.out > analysis_test.txt" 


# Print build type for verification
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

message(STATUS "ENABLE_CUDA: ${ENABLE_CUDA}")
if(ENABLE_CUDA)
        message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
endif()


#Setting the base version to C++ 17
set(CMAKE_CXX_STANDARD 17)

if(PERFORMANCE_METRICS)
        message("-- Setting PEREFORMANCE_METRICS: ON")
        add_definitions(-DPERFORMANCE_METRICS)
endif()

if(VALIDATION_MODE)
        message("-- Setting VALIDATION_MODE: ON")
        add_definitions(-DVALIDATION_MODE)
endif()

#HDF5 Support, finds HDF5 package for C and C++ and links the hdf5 libraries to the executable     \
#                  later in the file.
find_package(HDF5 COMPONENTS C CXX)
if (HDF5_FOUND)
        message("-- HDF5 version ${HDF5_VERSION} located. HDF5 recorders are available.")
        link_directories( ${HDF5_LIBRARY_DIRS} )
        include_directories( ${HDF5_INCLUDE_DIRS} )
        add_compile_definitions(HDF5)
else()
        message("-- HDF5 library was not located. Please only use XML recorders.")
endif()

#Find boost graph library
find_package(Boost REQUIRED COMPONENTS graph)

#Get git commit ID
#Change to "git rev-parse --short HEAD" for short commit ID
execute_process(
        COMMAND
        git rev-parse HEAD
        OUTPUT_VARIABLE        GIT_COMMIT_ID )

#GIT_COMMIT_ID has trailing whitespaces
string(REGEX REPLACE "\n$" "" GIT_COMMIT_ID "${GIT_COMMIT_ID}")

#Save GIT_COMMIT_ID to config.h, which is included in Core.cpp
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
                ${CMAKE_BINARY_DIR}/config.h
                @ONLY )

set(CMAKE_INCLUDE_CURRENT_DIR ON)

#set(CMAKE_VERBOSE_MAKEFILE TRUE)

#Setting the location of the executable to be in the top                                           \
#  - level directory.This helps when using file paths during
#runtime.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

#Set extra warning flags
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

#define TIXML_USE_STL as a preproccersser macro to use the C++ standard library with TinyXML
add_compile_definitions(TIXML_USE_STL)
message("-- Setting Compile Definition: TIMXL_USE_STL")

#Used to define file paths for #include definitions.For example, you can write:
# #include "Simulator.h"
#rather than:
# #include "Simulator/Core/Simulator.h"
#if the file path "Simulator/Core" is in the following statement.
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
        ThirdParty/log4cplus-2.0.7/include)
       
if(ENABLE_CUDA)
        
        set(cuda_ConnectionsSources 
        Simulator/Connections/Neuro/ConnGrowth_d.cpp )
        set_source_files_properties(${cuda_ConnectionsSources} PROPERTIES LANGUAGE CUDA)

        set(cuda_EdgesSources Simulator/Edges/Neuro/AllSynapsesDeviceFuncs_d.cpp
        Simulator/Edges/Neuro/AllSpikingSynapses_d.cpp
        Simulator/Edges/Neuro/AllSTDPSynapses_d.cpp
        Simulator/Edges/Neuro/AllDSSynapses_d.cpp
        Simulator/Edges/Neuro/AllDynamicSTDPSynapses_d.cpp
        Simulator/Edges/NG911/All911Edges_d.cpp)
        set_source_files_properties(${cuda_EdgesSources} PROPERTIES LANGUAGE CUDA)

        set(cuda_VerticesSources 
        Simulator/Vertices/Neuro/AllVerticesDeviceFuncs_d.cpp
        Simulator/Vertices/Neuro/AllLIFNeurons_d.cpp
        Simulator/Vertices/Neuro/AllIZHNeurons_d.cpp
        Simulator/Vertices/Neuro/AllIFNeurons_d.cpp
        Simulator/Vertices/Neuro/AllSpikingNeurons_d.cpp)
        set_source_files_properties(${cuda_VerticesSources} PROPERTIES LANGUAGE CUDA)
endif()

   #Collect source files and create libraries
   #Libraries containing.cpp files compiled as CUDA.cu files should be created as STATIC to ensure \
   #proper linking


   #Create Connections library
file(GLOB_RECURSE Connections_Source Simulator/Connections/*.cpp Simulator/Connections/*.h)

if(ENABLE_CUDA)
        add_library(Connections STATIC ${Connections_Source} ${cuda_ConnectionsSources})
else()
        list(REMOVE_ITEM Connections_Source "${CMAKE_CURRENT_SOURCE_DIR}/Simulator/Connections/Neuro/ConnGrowth_d.cpp")
        add_library(Connections STATIC ${Connections_Source})
endif()


# Create Vertices library
file(GLOB_RECURSE Vertices_Source Simulator/Vertices/*.cpp Simulator/Vertices/*.h)

if(ENABLE_CUDA)
        add_library(Vertices STATIC ${Vertices_Source} ${cuda_VerticesSources})
else()
        list(REMOVE_ITEM Vertices_Source "${CMAKE_CURRENT_SOURCE_DIR}/Simulator/Vertices/Neuro/AllVerticesDeviceFuncs.h")
        list(REMOVE_ITEM Vertices_Source "${CMAKE_CURRENT_SOURCE_DIR}/Simulator/Vertices/Neuro/AllVerticesDeviceFuncs_d.cpp")
        list(REMOVE_ITEM Vertices_Source "${CMAKE_CURRENT_SOURCE_DIR}/Simulator/Vertices/Neuro/AllLIFNeurons_d.cpp")
        list(REMOVE_ITEM Vertices_Source "${CMAKE_CURRENT_SOURCE_DIR}/Simulator/Vertices/Neuro/AllIZHNeurons_d.cpp")
        list(REMOVE_ITEM Vertices_Source "${CMAKE_CURRENT_SOURCE_DIR}/Simulator/Vertices/Neuro/AllIFNeurons_d.cpp")
        list(REMOVE_ITEM Vertices_Source "${CMAKE_CURRENT_SOURCE_DIR}/Simulator/Vertices/Neuro/AllSpikingNeurons_d.cpp")
        list(REMOVE_ITEM Vertices_Source "${CMAKE_CURRENT_SOURCE_DIR}/Simulator/Vertices/NG911/All911Vertices_d.cpp")
        
        add_library(Vertices STATIC ${Vertices_Source})
        
endif()


# Create Edges library
file(GLOB_RECURSE Edges_Source Simulator/Edges/*.cpp Simulator/Edges/*.h)
if(ENABLE_CUDA)
        add_library(Edges STATIC ${Edges_Source} ${cuda_EdgesSources})
else()
        list(REMOVE_ITEM Edges_Source "${CMAKE_CURRENT_SOURCE_DIR}/Simulator/Edges/Neuro/AllSynapsesDeviceFuncs.h")
        list(REMOVE_ITEM Edges_Source "${CMAKE_CURRENT_SOURCE_DIR}/Simulator/Edges/Neuro/AllSynapsesDeviceFuncs_d.cpp")
        list(REMOVE_ITEM Edges_Source "${CMAKE_CURRENT_SOURCE_DIR}/Simulator/Edges/Neuro/AllSynapsesDeviceFuncs_d_Growth.cpp")
        list(REMOVE_ITEM Edges_Source "${CMAKE_CURRENT_SOURCE_DIR}/Simulator/Edges/Neuro/AllSpikingSynapses_d.cpp")
        list(REMOVE_ITEM Edges_Source "${CMAKE_CURRENT_SOURCE_DIR}/Simulator/Edges/Neuro/AllSTDPSynapses_d.cpp")
        list(REMOVE_ITEM Edges_Source "${CMAKE_CURRENT_SOURCE_DIR}/Simulator/Edges/Neuro/AllDSSynapses_d.cpp")
        list(REMOVE_ITEM Edges_Source "${CMAKE_CURRENT_SOURCE_DIR}/Simulator/Edges/Neuro/AllDynamicSTDPSynapses_d.cpp")
        list(REMOVE_ITEM Edges_Source "${CMAKE_CURRENT_SOURCE_DIR}/Simulator/Edges/NG911/All911Edges_d.cpp")
        
        add_library(Edges STATIC ${Edges_Source})
        
endif()
# The Vertices library has a dependency on the Edges library
target_link_libraries(Vertices Edges)


# Create Layouts Library
file(GLOB_RECURSE Layouts_Source Simulator/Layouts/*.cpp Simulator/Layouts/*.h)
add_library(Layouts ${Layouts_Source})


# Create FunctionNodes library
file(GLOB FunctionNodes_Source Simulator/Core/FunctionNodes/*.cpp Simulator/Core/FunctionNodes/*.h)
add_library(FunctionNodes ${FunctionNodes_Source})


# Create Core library
file(GLOB Core_Source Simulator/Core/*.cpp Simulator/Core/*.h)
# Remove Graphitti_Main from the list of files, don't want 'main' in the library
list(REMOVE_ITEM Core_Source "${CMAKE_CURRENT_SOURCE_DIR}/Graphitti_Main.cpp")
# add_library(Core STATIC ${Core_Source})

if(ENABLE_CUDA)
        set(cudaCore_sources Simulator/Core/GPUModel.cpp)
        set_source_files_properties(${cudaCore_sources} PROPERTIES LANGUAGE CUDA)
        add_library(Core STATIC ${Core_Source} ${cudaCore_sources})
else()
        list(REMOVE_ITEM Core_Source "${CMAKE_CURRENT_SOURCE_DIR}/Simulator/Core/GPUModel.cpp")
        add_library(Core STATIC ${Core_Source})
endif()     


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
file(GLOB Utils_Source Simulator/Utils/*.cpp Simulator/Utils/*.h)
list(REMOVE_ITEM Utils_Source "${CMAKE_CURRENT_SOURCE_DIR}/Simulator/Utils/Factory.cpp")

if(CMAKE_BUILD_TYPE STREQUAL "Profiling")
        if(ENABLE_CUDA)
# Find NVTX Library
                find_library(NVTX_LIBRARY nvToolsExt)
                if(NVTX_LIBRARY)
                        message(STATUS "Found NVTX: ${NVTX_LIBRARY} included in Profiling")
                        add_compile_definitions(ENABLE_NVTX) 
                else()
                        message(STATUS "NVTX library not found! Not included in Profiling.")
                        list(REMOVE_ITEM Utils_Source "${CMAKE_CURRENT_SOURCE_DIR}/Simulator/Utils/NvtxHelper.cpp")
                endif()
        endif()

else()
        list(REMOVE_ITEM Utils_Source "${CMAKE_CURRENT_SOURCE_DIR}/Simulator/Utils/NvtxHelper.cpp")
endif()

# Always create the Utils library (even if NVTX and CUDA are missing)
add_library(Utils ${Utils_Source})

# Only link NVTX if it was found
if(NVTX_LIBRARY)
        target_link_libraries(Utils PRIVATE ${NVTX_LIBRARY})
endif()


# Used to locate and run other CMakeLists.txt files from Third Party resources for further compilation of the project.
add_subdirectory(ThirdParty)

# Googletest subdirectories support
add_subdirectory(Testing/lib/googletest-master)
include_directories(Testing/lib/GoogleTest/googletest-master/googletest/include)

# Set CUDA_SEPERABLE_COMPILATION to ON for all libraries that contain .cpp files with device functions compiled as CUDA files
if(ENABLE_CUDA)
        set_property(TARGET Edges PROPERTY CUDA_STANDARD 17)
        set_property(TARGET Vertices PROPERTY CUDA_STANDARD 17)
        # Enables CUDA code in these libraries to be compiled into separate object files and then linked together
        set_property(TARGET Edges PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET Vertices PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET Core PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET Connections PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET RNG PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET Utils PROPERTY CUDA_SEPARABLE_COMPILATION ON)
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
        paramcontainer
        Boost::graph)

# Link HDF5 package libraries if to the combined library.
if (HDF5_FOUND)
        target_link_libraries(combinedLib INTERFACE ${HDF5_LIBRARIES})
endif()

# ------ SIMULATOR EXECUTABLE -------
# Add all files that aren't in a library and are needed to run
if(ENABLE_CUDA)
        # Add the ggraphitti GPU executable
        add_executable(ggraphitti Simulator/Core/Graphitti_Main.cpp)
        # Link the combined library into the 'graphitti' executable.
	target_link_libraries(ggraphitti combinedLib)
else()
        # Add the cgraphitti CPU executable
	add_executable(cgraphitti Simulator/Core/Graphitti_Main.cpp)
	# Link the combined library into the 'graphitti' executable.
	target_link_libraries(cgraphitti combinedLib)
endif()

# ------ TESTS EXECUTABLE ------
# Add the file that contains main (RunTests.cpp) and all test files. GoogleTest will only recognize them if they are
# included in the executable.
target_compile_options(gtest PRIVATE -Wno-error=maybe-uninitialized)
target_compile_options(gtest_main PRIVATE -Wno-error=maybe-uninitialized)

if(ENABLE_CUDA)
    set(cuda_TestSources
        Testing/UnitTesting/DeviceVectorTests.cpp)
    set_source_files_properties(${cuda_TestSources} PROPERTIES LANGUAGE CUDA)
endif()

add_executable(tests
        Testing/RunTests.cpp
        Testing/UnitTesting/OperationManagerTests.cpp
        Testing/UnitTesting/EdgeIndexMapTests.cpp
        Testing/UnitTesting/FunctionNodeTests.cpp
        Testing/UnitTesting/SimulatorTests.cpp
        Testing/UnitTesting/OperationManagerTestingClass.h
        Testing/UnitTesting/VerticesFactoryTests.cpp
        Testing/UnitTesting/ConnectionsFactoryTests.cpp
        Testing/UnitTesting/EdgesFactoryTests.cpp
        Testing/UnitTesting/LayoutFactoryTests.cpp
        Testing/UnitTesting/RecorderFactoryTests.cpp
        Testing/UnitTesting/InputManagerTests.cpp
        Testing/UnitTesting/RNGFactoryTests.cpp
	Testing/UnitTesting/GraphManagerTests.cpp
        Testing/Utils/ParameterManagerTests.cpp
        Testing/Utils/CircularBufferTests.cpp
        Testing/UnitTesting/EventBufferTests.cpp
        Testing/UnitTesting/XmlRecorderTests.cpp
        Testing/UnitTesting/Hdf5RecorderTests.cpp
        Testing/UnitTesting/DeviceVectorTests.cpp)

# Links the Googletest framework with the testing executable
target_link_libraries(tests gtest gtest_main)

# Link the combined library into the 'tests' executable.
target_link_libraries(tests combinedLib)

#------- SERIALIZATION TESTS --------------------
# Serialization tests must be executed separately, as each test requires the 
# simulation to run from start to finish. Running multiple simulations with 
# the same singleton instance results in a segmentation fault (SEG FAULT).

add_executable(serialFullTest
        Testing/RunTests.cpp
        Testing/UnitTesting/SerializationFullTest.cpp)

add_executable(serialFirstHalfTest
        Testing/RunTests.cpp
        Testing/UnitTesting/SerializationFirstHalfTest.cpp)

add_executable(serialSecondHalfTest
        Testing/RunTests.cpp
        Testing/UnitTesting/SerializationSecondHalfTest.cpp)

# Link the GoogleTest framework to each of the serial test executables.
target_link_libraries(serialFullTest gtest gtest_main)
target_link_libraries(serialFirstHalfTest gtest gtest_main)
target_link_libraries(serialSecondHalfTest gtest gtest_main)

# Link the combined library and filesystem support to the respective serial test executables.
target_link_libraries(serialFullTest stdc++fs)
target_link_libraries(serialFullTest combinedLib)

target_link_libraries(serialFirstHalfTest stdc++fs)
target_link_libraries(serialFirstHalfTest combinedLib)

target_link_libraries(serialSecondHalfTest stdc++fs)
target_link_libraries(serialSecondHalfTest combinedLib)

# commenting out serialFileAccessTest until issue-754 is resolved
# add_executable(serialFileAccessTest
#        Testing/RunTests.cpp
#        Testing/UnitTesting/SerializationFileAccessTest.cpp)

# Links the Googletest framework with the serialFileAccessTest executable
# target_link_libraries(serialFileAccessTest gtest gtest_main)

# Link the combined library into the 'serialFileAccessTest' executable.
# target_link_libraries(serialFileAccessTest combinedLib)

# Clear ENABLE_CUDA, PERFORMANCE_METRICS and GPROF from the cache so it's reset for subsequent builds
unset(ENABLE_CUDA CACHE)
unset(PERFORMANCE_METRICS CACHE)
unset(GPROF CACHE)
unset(CMAKE_BUILD_TYPE CACHE)
unset(NVTX_LIBRARY CACHE)
unset(TARGET_ARCH CACHE)
unset(VALIDATION_MODE CACHE)