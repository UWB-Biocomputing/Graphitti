# Get all header and source files from current directory
file(GLOB RNG ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.h *.cu)

# Remove demo from file list as it contains a main and it will cause compilation errors
list(REMOVE_ITEM RNG "${CMAKE_CURRENT_SOURCE_DIR}/MersenneTwister_demo.cu")

# Create a library from the list of files gathered
add_library(RNG ${RNG})

if (ENABLE_CUDA)
   set_property(TARGET RNG PROPERTY CUDA_ARCHITECTURES 35-virtual) 
endif()