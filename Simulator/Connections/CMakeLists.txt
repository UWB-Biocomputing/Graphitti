# Get all header and source files from current directory and all subdirectories
file(GLOB_RECURSE Connections *.cpp *.h *.cu)

# Create a library from all the header and source files and in the current directory
add_library(Connections STATIC ${Connections})

if (ENABLE_CUDA)
   set_property(TARGET Connections PROPERTY LANGUAGE CUDA)
   set_property(TARGET Connections PROPERTY CUDA_ARCHITECTURES 35-virtual) 
   set_property(TARGET Connections PROPERTY LINKER_LANGUAGE CUDA)
endif()