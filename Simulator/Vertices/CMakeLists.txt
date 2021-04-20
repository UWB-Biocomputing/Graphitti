# Get all header and source files from current directory and all subdirectories
file(GLOB_RECURSE Vertices *.cpp *.h *.cu)

# Create a library from the list of files gathered
add_library(Vertices ${Vertices})

# The Vertices library has a dependency on the Edges library
target_link_libraries(Vertices Edges)

if (ENABLE_CUDA)
   set_property(TARGET Vertices PROPERTY CUDA_ARCHITECTURES 35-virtual) 
endif()