cmake_minimum_required(VERSION 3.10)
project(SummerOfBrain)

set(CMAKE_CXX_STANDARD 14)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Add all directories that contain required classes
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
        # Simulator Core
        Simulation/Core/OperationManager.h
        Simulation/Core/OperationManager.cpp
        Simulation/Core/Operations.h
        Simulation/Core/SynapseIndexMap.h
        Simulation/Core/Model.h
        Simulation/Core/Model.cpp
        Simulation/Core/Simulator.h
        Simulation/Core/Simulator.cpp
        Simulation/Core/FunctionNodes/GenericFunctionNode.h
        Simulation/Core/FunctionNodes/GenericFunctionNode.cpp

        # Simulator Connections
        Simulation/Connections/Connections.h
        Simulation/Connections/Connections.cpp
        Simulation/Connections/ConnGrowth.h
        Simulation/Connections/ConnGrowth.cpp
        Simulation/Connections/ConnStatic.h
        Simulation/Connections/ConnStatic.cpp

        # Simulator Layouts
        Simulation/Layouts/Layout.h
        Simulation/Layouts/Layout.cpp
        Simulation/Layouts/DynamicLayout.h
        Simulation/Layouts/DynamicLayout.cpp
        Simulation/Layouts/FixedLayout.h
        Simulation/Layouts/FixedLayout.cpp

        # Simulator Vertices
        Simulation/Vertices/IAllNeurons.h
        Simulation/Vertices/AllNeurons.h
        Simulation/Vertices/AllNeurons.cpp
        Simulation/Vertices/AllIFNeurons.h
        Simulation/Vertices/AllIFNeurons.cpp
        Simulation/Vertices/AllIZHNeurons.h
        Simulation/Vertices/AllIFNeurons.cpp
        Simulation/Vertices/AllLIFNeurons.h
        Simulation/Vertices/AllLIFNeurons.cpp
        Simulation/Vertices/AllNeuronsDeviceFuncs.h
        Simulation/Vertices/AllSpikingNeurons.h
        Simulation/Vertices/AllSpikingNeurons.cpp


        # Simulator Edges
        Simulation/Edges/AllDSSynapses.h
        Simulation/Edges/AllDSSynapses.cpp
        Simulation/Edges/AllDynamicSTDPSynapses.h
        Simulation/Edges/AllDynamicSTDPSynapses.cpp
        Simulation/Edges/AllSpikingSynapses.h
        Simulation/Edges/AllSpikingSynapses.cpp
        Simulation/Edges/AllSTDPSynapses.h
        Simulation/Edges/AllSTDPSynapses.cpp
        Simulation/Edges/AllSynapses.h
        Simulation/Edges/AllSynapses.cpp
        Simulation/Edges/AllSynapsesDeviceFuncs.h
        Simulation/Edges/IAllSynapses.h

        # Simulator Recorders
#        Simulation/Recorders/Hdf5GrowthRecorder.h
#        Simulation/Recorders/Hdf5GrowthRecorder.cpp
#        Simulation/Recorders/Hdf5Recorder.h
#        Simulation/Recorders/Hdf5Recorder.cpp
        Simulation/Recorders/IRecorder.h
        Simulation/Recorders/XmlGrowthRecorder.h
        Simulation/Recorders/XmlGrowthRecorder.cpp
        Simulation/Recorders/XmlRecorder.h
        Simulation/Recorders/XmlRecorder.cpp

        # Util Files
        Utils/BGTypes.h
        Utils/Global.h
        Utils/Global.cpp
        Utils/Timer.h
        Utils/Timer.cpp

        # Matrix
        Utils/Matrix/Matrix.h
        Utils/Matrix/Matrix.cpp
        Utils/Matrix/CompleteMatrix.h
        Utils/Matrix/CompleteMatrix.cpp
        Utils/Matrix/VectorMatrix.h
        Utils/Matrix/VectorMatrix.cpp

        # MersenneTwister
        Utils/RNG/MersenneTwister.h
        Utils/RNG/MersenneTwister.cpp
        Utils/RNG/Norm.h
        Utils/RNG/Norm.cpp

        # Test Files
        Testing/RunTests.cpp
        Testing/Core/Foo.h
        Testing/Core/Foo.cpp
        Testing/Core/FunctionNodeTests.cpp
        Testing/Core/OperationManagerTests.cpp
        Testing/Core/SynapseIndexMapTests.cpp
        Testing/Core/SimulatorTests.cpp)

# Links the Googletest framework with the testing executable
target_link_libraries(Tests gtest gtest_main)
